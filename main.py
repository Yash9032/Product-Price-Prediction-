import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# -------------------------------
# Load Dataset
# -------------------------------
d = pd.read_csv("updated_dataset_with_cost_price.csv")

# Independent & dependent columns
X = d.drop(columns=["date", "cost_price", "price"])
y = d["price"]

# Label Encoding for categorical columns
lb = LabelEncoder()
for col in ["product_id", "category", "region"]:
    X[col] = lb.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

# Polynomial transformation
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

# Accuracy
accuracy = r2_score(y_test, lr.predict(X_test_poly)) * 100

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("💰 Product Price Prediction App")

    # st.write(f"Model Accuracy: **{accuracy:.2f}%**")

    st.subheader("Enter Product Details:")

    product_id = st.selectbox("Product ID", sorted(d["product_id"].unique()))
    category = st.selectbox("Category", sorted(d["category"].unique()))
    region = st.selectbox("Region", sorted(d["region"].unique()))

    sales = st.number_input("Sales", min_value=0, value=10)
    discount = st.number_input("Discount", min_value=0.0, value=1.0)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.0)
    views = st.number_input("Views", min_value=0, value=100)
    clicks = st.number_input("Clicks", min_value=0, value=50)

    if st.button("Predict Price"):
        # Prepare input df
        input_data = pd.DataFrame({
            "product_id": [product_id],
            "sales": [sales],
            "category": [category],
            "region": [region],
            "discount": [discount],
            "rating": [rating],
            "views": [views],
            "clicks": [clicks]
        })

        # Encode categorical
        for col in ["product_id", "category", "region"]:
            input_data[col] = lb.fit_transform(input_data[col])

        # Transform using polynomial features
        input_poly = poly.transform(input_data)

        # Predict
        pred_price = lr.predict(input_poly)[0]

        st.success(f"✅ Predicted Price: **₹{pred_price:.2f}**")

if __name__ == "__main__":
    main()
