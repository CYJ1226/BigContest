import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from dateutil.relativedelta import relativedelta

class Feature_Engineering:

    def __init__(self, merchant_df, sale_df, cust_df, cm_ay1_df,cm_ay2_df,sales2023_df,sales2024_df,prediction_month):
        self.merchant_df = merchant_df.copy()
        self.sale_df = sale_df.copy()
        self.cust_df = cust_df.copy()
        self.cm_ay1_df = cm_ay1_df.copy()
        self.cm_ay2_df = cm_ay2_df.copy()
        self.sales2023_df = sales2023_df.copy()
        self.sales2024_df = sales2024_df.copy()
        self.prediction_month = prediction_month
        
        self.merchant_df_prc = None
        self.merged_df_prc = None
        self.total_df = None

    def rename_column(self):
        # merchant_df 컬럼명 변경
        self.merchant_df.columns = ['가맹점 구분번호', '가맹점 주소', '가맹점명', '브랜드 구분코드', '가맹점 지역', 
                                    '업종', '상권', '개업일', '폐업일','경도(lon)','위도(lat)','행정동_코드_명']

        # sale_df 컬럼명 변경
        self.sale_df.columns = ['가맹점 구분번호', '기준년월', '가맹점 운영개월수 구간', '매출금액 구간', '매출건수 구간', '유니크 고객 수 구간', '객단가 구간', '취소율 구간', '배달매출금액 비율', 
                        '동일 업종 매출금액 비율', '동일 업종 매출건수 비율', '동일 업종 내 매출 순위 비율', '동일 상권 내 매출 순위 비율', '동일 업종 내 해지 가맹점 비중', '동일 상권 내 해지 가맹점 비중']
        # cust_df 컬럼명 변경
        self.cust_df.columns = ['가맹점 구분번호', '기준년월', '남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중', '남성 50대 고객 비중', '남성 60대이상 고객 비중', '여성 20대이하 고객 비중', 
                        '여성 30대 고객 비중', '여성 40대 고객 비중', '여성 50대 고객 비중', '여성 60대이상 고객 비중','재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', '직장 이용 고객 비율', '유동인구 이용 고객 비율']
        return self

    def preprocess_merchant(self):
        
        merchant_df_prc = self.merchant_df.copy()

        merchant_df_prc['개업일'] = pd.to_datetime(merchant_df_prc['개업일'], format="%Y%m%d")
        merchant_df_prc['폐업일'] = pd.to_datetime(merchant_df_prc['폐업일'], format="%Y%m%d")


        #서울 성동구에 해당하지 않는 데이터 삭제
        merchant_df_prc = merchant_df_prc[merchant_df_prc['가맹점 주소'] != '경기 동두천시 중앙로265번길 30.']
        merchant_df_prc = merchant_df_prc[merchant_df_prc['가맹점 주소'] != '서울 강남구 선릉로 711.']                               
                
        # 업종 그룹화

        #한식음식점
        k_food = ['백반/가정식','기사식당','한식-두부요리','한식-단품요리일반','한정식','한식-죽', 
                  '한식-국수/만두','한식-국밥/설렁탕', '한식-찌개/전골','한식-냉면','한식뷔페',
                  '한식-감자탕','한식-해물/생선','한식-육류/고기','도시락']
        #중식음식점
        c_food = ['중식당','중식-딤섬/중식만두','중식-훠궈/마라탕']
        #일식음식점
        j_food = ['일식당','일식-우동/소바/라면','일식-초밥/롤','일식-덮밥/돈가스','일식-샤브샤브','일식-참치회']
        #양식음식점
        w_food = ['양식','스테이크']
        #제과점
        bakery = ['도너츠','마카롱','베이커리','와플/크로플','떡/한과','떡/한과 제조']
        #패스트푸드점
        fast_food = ['햄버거','피자','샌드위치/토스트']
        #치킨전문점
        chicken = ['치킨']
        #분식전문점
        convenience = ['분식']
        #호프-간이주점
        drink = ['호프/맥주','요리주점','민속주점','포장마차', '이자카야','와인바','일반 유흥주점','룸살롱/단란주점']
        #커피-음료
        cafe = ['카페','주스','차','테마카페','커피전문점','테이크아웃커피','아이스크림/빙수']
        #주류도매
        drink_sell = ['주류','와인샵']
        #미곡판매
        grain = ['농산물','미곡상']
        #육류판매
        meat = ['축산물']
        #수산물판매
        seafood = ['수산물','건어물']
        #청과상
        greengrocer = ['청과물']
        #반찬가게
        side_dish = ['반찬']
        #기타
        others = ['동남아/인도음식','기타세계요리','꼬치구이']
        nulls = ['유제품', '인삼제품', '건강식품', '건강원', '담배', '식료품','구내식당/푸드코트','탕후루'] #이 항목에 포함된 가맹점은 제외

        # ['샌드위치/토스트','식품 제조',  '유제품', '동남아/인도음식',  '기타세계요리',
                  #'인삼제품', '건강식품', '건강원', '담배', '식료품','구내식당/푸드코트','건어물','탕후루','떡/한과', '떡/한과 제조','꼬치구이']
        
        
        groups_to_replace = [ (k_food, '한식음식점'),(c_food, '중식음식점'),(j_food, '일식음식점'),(w_food, '양식음식점'),(bakery,'제과점'),(fast_food,'패스트푸드점'),
                              (chicken,'치킨전문점'),(convenience,'분식전문점'),(drink,'호프-간이주점'),(cafe, '커피-음료'),(drink_sell, '주류도매'),
                              (grain, '미곡판매'),(meat, '육류판매'),(seafood, '수산물판매'),(greengrocer, '청과상'),(side_dish, '반찬가게'),(others, '기타'),(nulls,np.nan)]
                            
        
        replacement = {i: cat for ind, cat in groups_to_replace for i in ind}
        merchant_df_prc['업종'].replace(replacement, inplace=True)
        merchant_df_prc.rename(columns = {'업종':'서비스_업종_코드_명'}, inplace=True)
        merchant_df_prc.replace('금호2.3가동','금호2·3가동',inplace=True)

 
        # 프랜차이즈 가맹점 구분
        merchant_df_prc['프랜차이즈'] = merchant_df_prc['브랜드 구분코드'].notnull().astype(int)

        merchant_df_prc.drop(columns=['가맹점 주소','가맹점명','브랜드 구분코드','상권','가맹점 지역','경도(lon)','위도(lat)'], inplace=True)
        self.merchant_df_prc = merchant_df_prc
        return self

    def preprocess_merged(self):
        if self.sale_df is None or self.cust_df is None:
            raise ValueError("매출, 고객 데이터가 로드되지 않았습니다.")

        merged_df_prc = self.sale_df.merge(self.cust_df, on=['가맹점 구분번호', '기준년월'])
        
        # 구간 데이터 숫자 추출
        band_cols_to_split = ['가맹점 운영개월수 구간', '매출금액 구간', '매출건수 구간', '유니크 고객 수 구간', '객단가 구간', '취소율 구간']
        for col in band_cols_to_split:
            merged_df_prc[col] = merged_df_prc[col].str.split('_', expand=True)[0]

        # -999999.9 값을 NaN으로 변경
        merged_df_prc.replace(-999999.9, np.nan, inplace=True)
        
        # 숫자형으로 변환
        numeric_cols = ['가맹점 운영개월수 구간', '매출금액 구간', '매출건수 구간', '유니크 고객 수 구간', '객단가 구간', 
                        '배달매출금액 비율', '동일 상권 내 해지 가맹점 비중']
        merged_df_prc[numeric_cols] = merged_df_prc[numeric_cols].astype(float)

        #결측치 처리
        merged_df_prc['배달매출금액 비율'] = merged_df_prc['배달매출금액 비율'].fillna(0)
        merged_df_prc['동일 상권 내 해지 가맹점 비중'] = merged_df_prc['동일 상권 내 해지 가맹점 비중'].fillna(0)
        
        # 방문 비율 100%로 정규화
        visit_total = merged_df_prc['재방문 고객 비중'] + merged_df_prc['신규 고객 비중']

        # 0으로 나누는 것을 방지
        visit_total[visit_total == 0] = 1 
        merged_df_prc['재방문 고객 비중'] = 100 * merged_df_prc['재방문 고객 비중'] / visit_total
        merged_df_prc['신규 고객 비중'] = 100 * merged_df_prc['신규 고객 비중'] / visit_total
    
        #기준년월 날짜형으로 변환
        merged_df_prc['기준년월'] = pd.to_datetime(merged_df_prc['기준년월'], format="%Y%m")
        

        merged_df_prc.drop(columns=['취소율 구간'], inplace=True)
        self.merged_df_prc = merged_df_prc
        return self

    def make_prediction_target(self):
        if self.merged_df_prc is None or self.merchant_df_prc is None:
            raise ValueError("선행 전처리 메서드(preprocess_merchant, preprocess_merged)를 먼저 실행해주세요.")
        
        simple_mc_df = self.merchant_df_prc[['가맹점 구분번호', '폐업일','개업일']]
        total_df = self.merged_df_prc.merge(simple_mc_df, on='가맹점 구분번호')
        total_df = total_df.sort_values(by=['가맹점 구분번호', '기준년월']).reset_index(drop=True)

        # 고객 관련 컬럼 등 일부 컬럼 제거
        cols_to_drop = ['남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중', '남성 50대 고객 비중', 
                        '남성 60대이상 고객 비중', '여성 20대이하 고객 비중', '여성 30대 고객 비중', '여성 40대 고객 비중', 
                        '여성 50대 고객 비중', '여성 60대이상 고객 비중', '재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', 
                        '직장 이용 고객 비율', '유동인구 이용 고객 비율', '동일 업종 내 해지 가맹점 비중', '동일 상권 내 해지 가맹점 비중', 
                        '동일 상권 내 매출 순위 비율','배달매출금액 비율', '가맹점 운영개월수 구간']
        total_df.drop(columns=cols_to_drop, inplace=True)

        # 폐업 예측 타겟 생성
        total_df['폐업 예측'] = total_df.apply(
            lambda row: 1 if pd.notna(row['폐업일']) and (row['기준년월'] < row['폐업일'] <= row['기준년월'] + pd.DateOffset(months=self.prediction_month)) else 0,
            axis=1)
        
        self.total_df = total_df
        return self

    def outlier_remove(self,percent = 0.995):
        upper_bound1 = self.total_df['동일 업종 매출금액 비율'].quantile(percent)
        upper_bound2 = self.total_df['동일 업종 매출건수 비율'].quantile(percent)

        final_df = self.total_df[(self.total_df['동일 업종 매출금액 비율']<=upper_bound1)&(self.total_df['동일 업종 매출건수 비율']<=upper_bound2)]
        self.total_df = final_df
        return self

    def select_merchants_by_status(self,data_select='part'):
        #폐업한 가계들 중 폐업 징조가 없는 월별 데이터 추출
        if self.total_df is None:
            raise ValueError("make_prediction_target 메서드를 먼저 실행해주세요.")
        
        if data_select == 'part':
            oc_mc_df = self.total_df[(self.total_df['폐업일'].notnull()) & (self.total_df['폐업 예측'] == 0)]
        elif data_select == 'all':
            oc_mc_df = self.total_df[self.total_df['폐업 예측'] == 0]
        elif data_select == 'not':
            oc_mc_df = self.total_df[self.total_df['폐업일'].isnull()]
        elif data_select == 'base':
            oc_mc_df = self.total_df[self.total_df['폐업일'].notnull()]
        else:
            raise ValueError("정확한 학습 데이터 조건을 입력해주세요.")
        
        oc_mc_list = oc_mc_df['가맹점 구분번호'].unique()
        self.total_df = self.total_df[self.total_df['가맹점 구분번호'].isin(oc_mc_list)].reset_index(drop=True)
        return self
    
    def make_feature(self):
        self.total_df["영업 개월"] =(12*(self.total_df['기준년월'].dt.year - self.total_df['개업일'].dt.year)+ 
                                 (self.total_df['기준년월'].dt.month - self.total_df['개업일'].dt.month))
        
        self.total_df["영업 개월"] = self.total_df["영업 개월"].astype(int)
        self.total_df.drop(columns=['개업일'], inplace=True)
   
    def create_recent_ma(self, months=[3, 6],min_periods = 2):
            
            if self.total_df is None: 
                raise ValueError("make_prediction_target 메서드를 먼저 실행해주세요.")
            
            remove_list = ['가맹점 구분번호','기준년월','폐업일', '폐업 예측','영업 개월']
            cols_to_ma = [i for i in self.total_df.columns if i not in remove_list]
            df_copy = self.total_df.copy()

            for m in months:
                for col in cols_to_ma:
                    new_col_name_ma = f'최근 {m}개월 평균_{col}' #최근 3,6 개월의 매출 특성 평균 
                    new_col_name_mv = f'최근 {m}개월 변동성_{col}' #최근 3,6 개월의 매출 특성 변동성(표준편차)
                    ma_series = df_copy.groupby('가맹점 구분번호')[col].rolling(window=m, min_periods=min_periods).mean()
                    mv_series = df_copy.groupby('가맹점 구분번호')[col].rolling(window=m, min_periods=min_periods).std()
                    df_copy[new_col_name_ma] = ma_series.reset_index(level=0, drop=True)
                    df_copy[new_col_name_mv] = mv_series.reset_index(level=0, drop=True)

            self.total_df = df_copy
            return self
    
    def create_lag_features(self, lag_periods=[1, 3, 6, 12]):
        #1,3,6,12개월 전 데이터 추출(lag)
        if self.total_df is None:
            raise ValueError("make_prediction_target 메서드를 먼저 실행해주세요.")

        cols_to_lag = ['매출금액 구간', '매출건수 구간', '유니크 고객 수 구간', '객단가 구간']
        
        df_copy = self.total_df.copy()

        for col in cols_to_lag:
            for period in lag_periods:
                new_col_name = f'{period}개월전_{col}'
                df_copy[new_col_name] = df_copy.groupby('가맹점 구분번호')[col].shift(period)
        
        self.total_df = df_copy
        return self
    
    def add_merchant_data(self):
        # 최종 병합

        final_df = self.total_df.merge(self.merchant_df_prc, on='가맹점 구분번호', suffixes=('', '_y'))
        final_df['기준_년분기_코드'] = 10*(final_df['기준년월'].dt.year) + final_df['기준년월'].dt.quarter
 

        final_df = pd.merge(final_df,self.cm_ay1_df, on = ['행정동_코드_명','기준_년분기_코드','서비스_업종_코드_명'], how='left')
        final_df = pd.merge(final_df,self.cm_ay2_df, on = ['기준_년분기_코드','행정동_코드_명'], how='left')
 
        sales_col_list = ['기준_년분기_코드','행정동_코드_명','서비스_업종_코드_명','당월_매출_금액','당월_매출_건수']
        sales2023_df_prc = self.sales2023_df[sales_col_list].copy()
        sales2024_df_prc = self.sales2024_df[sales_col_list].copy()

        sales_total_df_prc = pd.concat([sales2023_df_prc, sales2024_df_prc], ignore_index=True)
        final_df = pd.merge(final_df,sales_total_df_prc, on=['행정동_코드_명', '기준_년분기_코드', '서비스_업종_코드_명'],how='left') 

        str2num = ['유동인구_명_ha','임대료']

        for col in str2num: #문자열 데이터를 숫자형 데이터로 변환
            if col in final_df.columns:
                final_df[col] = final_df[col].astype(str).str.replace(',', '').astype(float)


        final_df['당월 매출 금액 평균'] = final_df['당월_매출_금액']/final_df['전체 점포수']
        final_df['당월 매출 건수 평균'] = final_df['당월_매출_건수']/final_df['전체 점포수']

        self.final_df = final_df
        return self
    
    def run(self,percent = 0.995,data_select='part',months=[3,6],lag_periods=[1, 3, 6, 12],min_periods = 2):
        self.rename_column()
        self.preprocess_merchant()
        self.preprocess_merged()
        self.make_prediction_target()
        self.outlier_remove(percent = percent)
        self.select_merchants_by_status(data_select=data_select)
        self.make_feature()
        self.create_recent_ma(months=months,min_periods = min_periods)
        self.create_lag_features(lag_periods=lag_periods)
        self.add_merchant_data()
        
        #중복 컬럼 제거
        self.final_df.drop(columns=[col for col in self.final_df.columns if '_y' in col], inplace=True)
        self.final_df = self.final_df.drop(columns=['가맹점 구분번호', '기준년월', '폐업일','개업일','개업수','폐업수','일반 점포수','프랜차이즈 점포수','기준_년분기_코드','당월_매출_금액','당월_매출_건수'])
        self.final_df = self.final_df.dropna(subset = ['최근 3개월 평균_매출금액 구간','서비스_업종_코드_명'])

        numeric_cols = self.final_df.select_dtypes(include=np.number).columns
        self.final_df[numeric_cols] = self.final_df[numeric_cols].round(2)
        
        return self.final_df