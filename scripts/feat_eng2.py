import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

class Feature_Engineering:

    def __init__(self, merchant_df, sale_df, cust_df, prediction_month):
        self.merchant_df = merchant_df.copy()
        self.sale_df = sale_df.copy()
        self.cust_df = cust_df.copy()
        self.prediction_month = prediction_month
        
        self.merchant_df_prc = None
        self.merged_df_prc = None
        self.total_df = None

    def rename_column(self):
        # merchant_df 컬럼명 변경
        self.merchant_df.columns = ['가맹점 구분번호', '가맹점 주소', '가맹점명', '브랜드 구분코드', '가맹점 지역', '업종', '상권', '개업일', '폐업일']

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
                
        # 업종 그룹화
        meat = ['한식-육류/고기',  '꼬치구이']
        cafe = ['카페',  '주스',  '차',  '테마카페',  '커피전문점', '테이크아웃커피',  '구내식당/푸드코트']
        k_food = ['백반/가정식',  '기사식당', '한식-두부요리', '한식-단품요리일반',  '한정식',    '한식-죽',  '한식-국수/만두',  '한식-국밥/설렁탕',  '한식-찌개/전골',  '한식-냉면',  '한식뷔페',  '한식-감자탕',   '한식-해물/생선']
        w_food = ['양식',  '스테이크', '치킨',  '햄버거',  '피자']
        j_food = ['일식당',  '일식-우동/소바/라면',  '일식-초밥/롤',  '일식-덮밥/돈가스',  '일식-샤브샤브',  '일식-참치회']
        c_food = ['중식당',  '중식-딤섬/중식만두',  '중식-훠궈/마라탕']
        drink = ['호프/맥주',  '요리주점',  '민속주점',  '포장마차',  '이자카야',  '와인바', '주류',  '와인샵']
        product= ['농산물',  '청과물',  '수산물',  '건어물',  '축산물']
        enter = ['일반 유흥주점',  '룸살롱/단란주점']
        convenience = ['샌드위치/토스트',  '도시락', '분식']
        world_food = ['동남아/인도음식',  '기타세계요리']
        dessert = ['도너츠',  '탕후루',  '와플/크로플',  '마카롱',  '아이스크림/빙수',  '떡/한과',  '떡/한과 제조',  '베이커리']
        others = ['식품 제조',  '반찬',  '미곡상',  '유제품',  '인삼제품', '건강식품', '건강원', '담배',  '식료품']
        
        groups_to_replace = [(meat, '육류'), (cafe, '카페'), (k_food, '한식'), (w_food, '양식'), (j_food, '일식'),
                            (c_food, '중식'), (drink, '주점'), (product, '농수축산물'), (enter, '유흥업소'), (convenience, '간편식'),
                            (world_food, '이색요리'), (dessert, '디저트'), (others, '기타')]
        
        replacement = {i: cat for ind, cat in groups_to_replace for i in ind}
        merchant_df_prc['업종'].replace(replacement, inplace=True)

        # 상권 그룹화
        areas_to_replace = {'화양시장': '성수', '자양': '성수', '서면역': '성수', '미아사거리': '성수',
                            '방배역': '뚝섬', '건대입구': '뚝섬', '풍산지구': '뚝섬', '오남': '한양대',
                            '동대문역사문화공원역': '금남시장',  '압구정로데오': '금남시장',  '장한평자동차': '답십리'}
        
        merchant_df_prc['상권'].replace(areas_to_replace, inplace=True)
        merchant_df_prc['상권'].fillna('Unknown', inplace=True)

        ## 상권 결측치 대체
        merchant_df_prc.loc[merchant_df_prc['상권']=='Unknown', '가맹점 주소'].str.split(expand=True)[2].value_counts()
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로 410')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로31')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('마장로 137')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('무학로 33')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('마장로35길')), '상권'] = '마장동'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('마장로37길')), '상권'] = '마장동'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('마장')), '상권'] = '마장동'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('금호로')), '상권'] = '신금호'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('행당로')), '상권'] = '행당'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로 58')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 구분번호'].str.contains('1F0AADBBB8')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로 16')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로14')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('동일로')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('광나루로')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('아차산로')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('매봉길')), '상권'] = '옥수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('옥수2동')), '상권'] = '옥수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('연무장')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('성덕정')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('서울숲2길')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('무학봉길')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('송정')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('성수')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('둘레1길')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('독서당로')), '상권'] = '행당'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로 50')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리로 6')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('왕십리')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('무학')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('난계로')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('홍익동')), '상권'] = '왕십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('살곶이')), '상권'] = '한양대'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('마조로')), '상권'] = '한양대'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('사근동')), '상권'] = '한양대'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('뚝섬로 3')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('뚝섬로1가')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('뚝섬로1길')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('서울숲길')), '상권'] = '뚝섬'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('뚝섬로')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('선릉')), '상권'] = '성수'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('고산자로')), '상권'] = '행당'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('응봉동')), '상권'] = '행당'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('장터')), '상권'] = '금남시장'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('동호')), '상권'] = '금남시장'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('청계천')), '상권'] = '마장동'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('금호')), '상권'] = '신금호'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('자동차시장길')), '상권'] = '답십리'
        merchant_df_prc.loc[(merchant_df_prc['상권']=='Unknown') & (merchant_df_prc['가맹점 주소'].str.contains('용답')), '상권'] = '답십리'

        merchant_df_prc.drop(columns=['가맹점 주소','가맹점명','브랜드 구분코드','가맹점 지역','개업일'], inplace=True)
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
        
        simple_mc_df = self.merchant_df_prc[['가맹점 구분번호', '폐업일']]
        total_df = self.merged_df_prc.merge(simple_mc_df, on='가맹점 구분번호')
        total_df = total_df.sort_values(by=['가맹점 구분번호', '기준년월']).reset_index(drop=True)

        # 고객 관련 컬럼 등 일부 컬럼 제거
        cols_to_drop = ['남성 20대이하 고객 비중', '남성 30대 고객 비중', '남성 40대 고객 비중', '남성 50대 고객 비중', 
                        '남성 60대이상 고객 비중', '여성 20대이하 고객 비중', '여성 30대 고객 비중', '여성 40대 고객 비중', 
                        '여성 50대 고객 비중', '여성 60대이상 고객 비중', '재방문 고객 비중', '신규 고객 비중', '거주 이용 고객 비율', 
                        '직장 이용 고객 비율', '유동인구 이용 고객 비율', '동일 업종 내 해지 가맹점 비중', '동일 상권 내 해지 가맹점 비중', 
                        '배달매출금액 비율', '가맹점 운영개월수 구간']
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
            oc_mc_df = self.total_df
        elif data_select == 'not':
            oc_mc_df = self.total_df[self.total_df['폐업일'].isnull()]
        elif data_select == 'base':
            oc_mc_df = self.total_df[self.total_df['폐업일'].notnull()]
        else:
            raise ValueError("정확한 학습 데이터 조건을 입력해주세요.")
        
        oc_mc_list = oc_mc_df['가맹점 구분번호'].unique()
        self.total_df = self.total_df[self.total_df['가맹점 구분번호'].isin(oc_mc_list)].reset_index(drop=True)
        return self
   
    def create_recent_ma(self, months=[3, 6]):
            
            if self.total_df is None: 
                raise ValueError("make_prediction_target 메서드를 먼저 실행해주세요.")
            
            remove_list = ['가맹점 구분번호','기준년월','폐업일', '폐업 예측']
            cols_to_ma = [i for i in self.total_df.columns if i not in remove_list]
            df_copy = self.total_df.copy()

            for m in months:
                for col in cols_to_ma:
                    new_col_name = f'최근 {m}개월 평균_{col}'
                    ma_series = df_copy.groupby('가맹점 구분번호')[col].rolling(window=m, min_periods=2).mean()
                    df_copy[new_col_name] = ma_series.reset_index(level=0, drop=True)

                int_cols = df_copy.loc[:,f'최근 {m}개월 평균_'+'매출금액 구간':f'최근 {m}개월 평균_'+'객단가 구간']    
                for i in int_cols:
                    df_copy[i] = df_copy[i].round(0)

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
    
    def run(self,percent = 0.995,data_select='part',months=[3,6],lag_periods=[1, 3, 6, 12]):
        self.rename_column()
        self.preprocess_merchant()
        self.preprocess_merged()
        self.make_prediction_target()
        self.outlier_remove(percent = percent)
        self.select_merchants_by_status(data_select=data_select)
        self.create_recent_ma(months=months)
        self.create_lag_features(lag_periods=lag_periods)

        #최종 병합
        final_df = self.total_df.merge(self.merchant_df_prc, on='가맹점 구분번호', suffixes=('', '_y'))
        #중복 컬럼 제거
        final_df.drop(columns=[col for col in final_df.columns if '_y' in col], inplace=True)
        final_df = final_df.drop(columns=['가맹점 구분번호', '기준년월', '폐업일'])
        final_df = final_df.dropna(subset = ['최근 3개월 평균_매출금액 구간'])
        self.final_df = final_df
        return self.final_df