import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from fpdf import FPDF

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(10)

data=pd.DataFrame({
'Year':np.random.randint(2005,2023,500),
'Present_Price':np.random.uniform(2,25,500),
'Driven_kms':np.random.randint(5000,120000,500),
'Fuel_Type':np.random.choice(['Petrol','Diesel','CNG'],500),
'Seller_Type':np.random.choice(['Dealer','Individual'],500),
'Transmission':np.random.choice(['Manual','Automatic'],500),
'Owner':np.random.randint(0,3,500),
'Selling_Price':np.random.uniform(1,20,500)
})

current_year=2024
data['Car_Age']=current_year-data['Year']

data=pd.get_dummies(data,drop_first=True)

X=data.drop('Selling_Price',axis=1)
y=data['Selling_Price']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

models={
'Linear Regression':LinearRegression(),
'Ridge Regression':Ridge(),
'Lasso Regression':Lasso(),
'Random Forest':RandomForestRegressor(n_estimators=200,random_state=42)
}

results=[]

predictions={}

for name,model in models.items():

    if name=='Random Forest':
        model.fit(X_train,y_train)
        pred=model.predict(X_test)
    else:
        model.fit(X_train_scaled,y_train)
        pred=model.predict(X_test_scaled)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    results.append([name,mae,rmse,r2])
    predictions[name]=pred

results_df=pd.DataFrame(results,columns=['Model','MAE','RMSE','R2'])

results_path=os.path.join(output_dir,"model_results.csv")
results_df.to_csv(results_path,index=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Model',y='R2',data=results_df)
plt.title("Model Performance Comparison")
plt.xticks(rotation=30)
chart1=os.path.join(charts_dir,"model_comparison.png")
plt.tight_layout()
plt.savefig(chart1)
plt.close()

rf_model=models['Random Forest']
rf_model.fit(X_train,y_train)

importance=rf_model.feature_importances_

imp_df=pd.DataFrame({
'Feature':X.columns,
'Importance':importance
}).sort_values(by='Importance',ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance',y='Feature',data=imp_df.head(10))
plt.title("Feature Importance")
chart2=os.path.join(charts_dir,"feature_importance.png")
plt.tight_layout()
plt.savefig(chart2)
plt.close()

rf_pred=rf_model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test,rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
chart3=os.path.join(charts_dir,"actual_vs_predicted.png")
plt.tight_layout()
plt.savefig(chart3)
plt.close()

residuals=y_test-rf_pred

plt.figure(figsize=(6,4))
sns.histplot(residuals,bins=30,kde=True)
plt.title("Residual Distribution")
chart4=os.path.join(charts_dir,"residual_distribution.png")
plt.tight_layout()
plt.savefig(chart4)
plt.close()

prediction_table=pd.DataFrame({
'Actual_Price':y_test.values,
'Predicted_Price':rf_pred
})

table_path=os.path.join(output_dir,"prediction_table.csv")
prediction_table.to_csv(table_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Car Price Prediction Report",0,1,"C")

pdf.set_font("Arial","",12)

text="This report presents a machine learning based system for predicting used car prices using regression algorithms. Multiple regression models were trained and evaluated to determine the most accurate predictor."

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in results_df.iterrows():
    line=f"{row['Model']} | MAE: {row['MAE']:.2f} | RMSE: {row['RMSE']:.2f} | R2: {row['R2']:.2f}"
    pdf.cell(0,8,line,0,1)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Comparison Chart",0,1)
pdf.image(chart1,w=170)

pdf.ln(10)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Importance",0,1)
pdf.image(chart2,w=170)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Actual vs Predicted Prices",0,1)
pdf.image(chart3,w=170)

pdf.ln(10)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Residual Distribution",0,1)
pdf.image(chart4,w=170)

pdf_path=os.path.join(output_dir,"Car_Price_Prediction_Report.pdf")
pdf.output(pdf_path)

print(results_df)
print("\nAll outputs saved in:",output_dir)