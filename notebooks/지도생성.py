import pandas as pd
import json
import os
import http.server
import socketserver
import webbrowser

# 데이터 로드 및 전처리
df = pd.read_csv('../data/processed/행정동.csv')
df['is_closed']  = df['폐업일'].notna().astype(int)
df.rename({'가맹점구분번호': 'store_name', '위도(lat)': 'latitude', '경도(lon)': 'longitude'}, axis=1, inplace=True)
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# API KEY 및 변수 할당
JAVASCRIPT_KEY = "87119a9a672d25a5916aa93907429177" 
HTML_FILE_NAME = "store_locations_map.html"
PORT = 8000 # 로컬 서버 포트


# 지도를 위한 데이터 구조를 JavaScript JSON 문자열로 변환
store_list = df[['store_name', 'latitude', 'longitude', 'is_closed']].to_dict('records')
store_data_json = json.dumps(store_list, ensure_ascii=False)
num_stores = len(store_list)

# HTML 파일 생성 함수
def generate_bulk_map_html(store_data_json, api_key, num_stores):
    num_stores = len(store_list)
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>총 {num_stores}개 가맹점 위치</title>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            overflow: hidden; /* 스크롤바 제거 */
        }}
    </style>
</head>
<body>
    <div id="map" style="width:100%;height:100%;margin: 20px auto; border: none;"></div>
    
    <script type="text/javascript" src="https://dapi.kakao.com/v2/maps/sdk.js?appkey={api_key}&autoload=false&libraries=services"></script>
    <script>
        // Python에서 JSON으로 전달된 가맹점 데이터 ({store_data_json}은 Python에서 채워집니다)
        var storeData = {store_data_json};

        // 지도 초기화 및 마커 생성 메인 함수
        function initMap() {{
            
            // 마커 이미지 정의 (폐업 여부에 따른 이미지)

            // 일반 가맹점 마커 이미지 (파란색 마커)
            var normalImageSrc = "https://t1.daumcdn.net/mapjsapi/images/marker.png";
            var normalImageSize = new kakao.maps.Size(10, 16); 
            var normalMarkerImage = new kakao.maps.MarkerImage(normalImageSrc, normalImageSize);

            // 폐업 가맹점 마커 이미지 (빨간색 마커)
            var closedImageSrc = "red_marker.png";
            var closedImageSize = new kakao.maps.Size(30, 36);
            var closedMarkerImage = new kakao.maps.MarkerImage(closedImageSrc, closedImageSize);

            var centerLat, centerLng; 

            if (storeData.length === 0) {{
                centerLat = 37.566826;
                centerLng = 126.9786567;
            }} else {{
                // 데이터의 평균 위도/경도를 계산하여 중심 좌표 설정
                centerLat = storeData.reduce((sum, d) => sum + d.latitude, 0) / storeData.length;
                centerLng = storeData.reduce((sum, d) => sum + d.longitude, 0) / storeData.length;
            }}
            
            var mapContainer = document.getElementById('map');
            var mapOption = {{
                center: new kakao.maps.LatLng(centerLat, centerLng),
                level: 7
            }};

            // 지도 객체 생성
            var map = new kakao.maps.Map(mapContainer, mapOption);

            // 마커 생성 반복문
            for (var i = 0; i < storeData.length; i++) {{
                var store = storeData[i];

                // 폐업 여부 확인
                var markerIcon;
                if (store.is_closed === 1) {{
                    markerIcon = closedMarkerImage; // 폐업이면 빨간색 마커
                }} else {{
                    markerIcon = normalMarkerImage; // 정상이면 기본 마커
                }}

                var markerPosition = new kakao.maps.LatLng(store.latitude, store.longitude);

                // 마커 객체 생성 시, 결정된 이미지를 image 속성에 할당
                var marker = new kakao.maps.Marker({{
                    position: markerPosition,
                    title: store.store_name, 
                    image: markerIcon 
                }});

                // 지도에 마커 표시
                marker.setMap(map);
            }}
        }}

        // 지도 SDK 로드가 완료된 후 initMap 함수 실행
        kakao.maps.load(initMap);
    </script>
</body>
</html>
"""
    return html_template

# HTML 파일 생성 및 저장
html_output = generate_bulk_map_html(store_data_json, JAVASCRIPT_KEY, num_stores)

with open(HTML_FILE_NAME, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"지도 HTML 파일 생성 완료: {os.path.abspath(HTML_FILE_NAME)}")

# 로컬 서버 실행 및 브라우저 열기

Handler = http.server.SimpleHTTPRequestHandler

try:
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        server_url = f"http://127.0.0.1:{PORT}/{HTML_FILE_NAME}"
        
        # 브라우저 자동 실행
        webbrowser.open_new_tab(server_url)
        
        # 서버 유지
        httpd.serve_forever()

except Exception as e:
    print("서버 실행 중 오류 발생")