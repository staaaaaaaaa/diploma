from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd


def data_beautifier(df):
    X_encoded = pd.get_dummies(df, columns=['country', 'exchange', 'sector'], drop_first=True)
    normalizer = StandardScaler()
    X_real_norm_np = normalizer.fit_transform(X_encoded)
    X = pd.DataFrame(data=X_real_norm_np)
    X.columns = X_encoded.columns.tolist()
    return X

# def dprediction(X):

        
#         ebitPKL = pickle.load(open('ebitgrowth.pkl', 'rb'))
#         epsPKL = pickle.load(open('ebitgrowth.pkl', 'rb'))
#         netIncomePKL = pickle.load(open('ebitgrowth.pkl', 'rb'))
#         operatingIncomePKL = pickle.load(open('ebitgrowth.pkl', 'rb'))
#         revenuePKL = pickle.load(open('ebitgrowth.pkl', 'rb'))

#         c1, c2, c3, c4, c5 = st.columns(5)
#         c1.metric(label = 'Рост EBIT', value=ebitPKL.predict(user_df)[0])
#         c2.metric(label = 'Рост EPS', value=epsPKL.predict(user_df)[0])
#         c3.metric(label = 'Рост Чистого дохода', value=netIncomePKL.predict(user_df)[0])
#         c4.metric(label = 'Рост Оперативного дохода', value=operatingIncomePKL.predict(user_df)[0])
#         c5.metric(label = 'Рост  Выручки', value=revenuePKL.predict(user_df)[0])

