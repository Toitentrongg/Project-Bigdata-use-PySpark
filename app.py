from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Thêm dòng này để import CORS
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from datetime import datetime
import math

# Tạo ứng dụng Flask
app = Flask(__name__)
CORS(app)  # Thêm dòng này để áp dụng CORS cho toàn bộ ứng dụng

# Khởi tạo SparkSession
spark = SparkSession.builder \
    .appName("Taxi Fare Prediction") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.executor.memory", "12g") \
    .config("spark.driver.memory", "12g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()


# Đường dẫn tới mô hình đã lưu
model_path = "file:///C:/Model_Taxi/taxi_fare_rf_model"
# Tải mô hình
loaded_model = RandomForestRegressionModel.load(model_path)

# Hàm Haversine tính khoảng cách giữa hai điểm dựa trên kinh độ và vĩ độ
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Bán kính trái đất (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Tạo UDF (User Defined Function) để tính khoảng cách
haversine_udf = udf(haversine, DoubleType())

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    pickup_datetime_str = request.form['pickup_datetime']
    dropoff_datetime_str = request.form['dropoff_datetime']

    # Chuyển đổi chuỗi thành datetime
    pickup_datetime = datetime.strptime(pickup_datetime_str, '%Y-%m-%dT%H:%M')
    dropoff_datetime = datetime.strptime(dropoff_datetime_str, '%Y-%m-%dT%H:%M')

    # Tính toán trip_duration, pickup_hour, pickup_year
    trip_duration = (dropoff_datetime - pickup_datetime).total_seconds() / 60  # phút
    pickup_hour = pickup_datetime.hour
    pickup_year = pickup_datetime.year

    def is_valid_coordinates(longitude, latitude):
        if -180 <= longitude <= 180 and -90 <= latitude <= 90:
            return True
        return False

    # Lấy các giá trị khác từ form
    passenger_count = int(request.form['passenger_count'])
    trip_distance = float(request.form['trip_distance'])
    pickup_longitude = float(request.form['pickup_longitude'])
    pickup_latitude = float(request.form['pickup_latitude'])
    dropoff_longitude = float(request.form['dropoff_longitude'])
    dropoff_latitude = float(request.form['dropoff_latitude'])

    if not (is_valid_coordinates(pickup_longitude, pickup_latitude) and
            is_valid_coordinates(dropoff_longitude, dropoff_latitude)):
        return jsonify({'error': 'Invalid coordinates'}), 400

    # Tính toán distance_haversine
    distance_haversine = haversine(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

    # Tạo DataFrame từ dữ liệu người dùng
    input_data = Row(
        passenger_count=passenger_count,
        trip_distance=trip_distance,
        pickup_longitude=pickup_longitude,
        pickup_latitude=pickup_latitude,
        dropoff_longitude=dropoff_longitude,
        dropoff_latitude=dropoff_latitude,
        trip_duration=trip_duration,
        pickup_hour=pickup_hour,
        pickup_year=pickup_year,
        distance_haversine=distance_haversine
    )

    # Tạo DataFrame từ dữ liệu đầu vào
    input_df = spark.createDataFrame([input_data])

    # Tạo cột features
    feature_columns = ["passenger_count", "trip_distance", "pickup_longitude", "pickup_latitude",
                       "dropoff_longitude", "dropoff_latitude", "trip_duration",
                       "pickup_hour", "pickup_year", "distance_haversine"]

    # Tạo vector assembler và transform
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    input_df = vector_assembler.transform(input_df)

    # Dự đoán
    prediction = loaded_model.transform(input_df)
    predicted_fare = prediction.select("prediction").first()[0]

    return jsonify({'predicted_fare': predicted_fare})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

