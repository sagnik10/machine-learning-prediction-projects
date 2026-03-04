import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from fpdf import FPDF

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'Rainfall':np.random.randint(300,900,500),
'Temperature':np.random.randint(18,35,500),
'Humidity':np.random.randint(50,90,500),
'Soil_pH':np.round(np.random.uniform(5.5,7.5,500),2),
'Soil_Type':np.random.choice(['Loamy','Sandy','Clay'],500),
'Fertilizer':np.random.randint(1,4,500),
'Crop_Type':np.random.choice(['Rice','Wheat','Maize','Cotton','Sugarcane'],500),
'Yield':np.random.randint(1800,4500,500)
})

data=pd.get_dummies(data,drop_first=True)

X=data.drop('Yield',axis=1)
y=data['Yield']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
'Linear Regression':LinearRegression(),
'Ridge Regression':Ridge(),
'Random Forest':RandomForestRegressor(n_estimators=200,random_state=42)
}

results=[]
predictions={}

for name,model in models.items():

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    results.append([name,mae,rmse,r2])
    predictions[name]=pred

results_df=pd.DataFrame(results,columns=['Model','MAE','RMSE','R2'])

results_path=os.path.join(output_dir,"model_metrics.csv")
results_df.to_csv(results_path,index=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Model',y='R2',data=results_df)
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=20)
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
plt.title("Feature Importance (Random Forest)")
chart2=os.path.join(charts_dir,"feature_importance.png")
plt.tight_layout()
plt.savefig(chart2)
plt.close()

rf_pred=rf_model.predict(X_test)

plt.figure(figsize=(6,5))
plt.scatter(y_test,rf_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
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

corr=data.corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr,cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
chart5=os.path.join(charts_dir,"correlation_heatmap.png")
plt.tight_layout()
plt.savefig(chart5)
plt.close()

prediction_table=pd.DataFrame({
'Actual_Yield':y_test.values,
'Predicted_Yield':rf_pred
})

table_path=os.path.join(output_dir,"yield_predictions.csv")
prediction_table.to_csv(table_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Crop Yield Prediction Report",0,1,"C")

pdf.set_font("Arial","",12)

text="This report presents machine learning based crop yield prediction using regression models trained on agro-climatic parameters including rainfall, temperature, humidity, soil characteristics and fertilizer usage."

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
pdf.cell(0,10,"Feature Importance Analysis",0,1)
pdf.image(chart2,w=170)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Actual vs Predicted Yield",0,1)
pdf.image(chart3,w=170)

pdf.ln(10)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Residual Distribution",0,1)
pdf.image(chart4,w=170)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Correlation Heatmap",0,1)
pdf.image(chart5,w=170)

pdf_path=os.path.join(output_dir,"Crop_Yield_Prediction_Report.pdf")
pdf.output(pdf_path)

print(results_df)
print("\nOutputs saved in:",output_dir)