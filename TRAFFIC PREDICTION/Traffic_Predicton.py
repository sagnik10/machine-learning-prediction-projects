import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from fpdf import FPDF
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'Hour':np.random.randint(0,24,500),
'DayOfWeek':np.random.randint(0,7,500),
'Temperature':np.random.randint(18,35,500),
'Humidity':np.random.randint(40,90,500),
'Rainfall':np.random.randint(0,20,500),
'Event':np.random.randint(0,2,500),
'TrafficVolume':np.random.randint(200,900,500)
})

features=['Hour','DayOfWeek','Temperature','Humidity','Rainfall','Event']
target='TrafficVolume'

X=data[features]
y=data[target]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
"Linear Regression":LinearRegression(),
"Ridge Regression":Ridge(alpha=1.0),
"SVR":SVR(kernel='rbf'),
"Random Forest":RandomForestRegressor(n_estimators=200,random_state=42)
}

metrics=[]
results={}

for name,model in models.items():

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)

    metrics.append([name,mae,rmse,r2])

    result=pd.DataFrame({
        'Actual':y_test.values,
        'Predicted':pred
    })

    results[name]=result

    plt.figure(figsize=(6,4))
    plt.plot(result['Actual'].values,label="Actual",marker='o')
    plt.plot(result['Predicted'].values,label="Predicted",marker='x')
    plt.title(name+" Traffic Prediction")
    plt.xlabel("Samples")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,name.replace(" ","_")+"_prediction.png")
    plt.savefig(chart_path,dpi=300,bbox_inches="tight")
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Model','MAE','RMSE','R2'])

plt.figure(figsize=(8,4))
plt.bar(metrics_df['Model'],metrics_df['RMSE'])
plt.title("RMSE Comparison of Models")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()

rmse_chart=os.path.join(charts_dir,"rmse_comparison.png")
plt.savefig(rmse_chart,dpi=300,bbox_inches="tight")
plt.close()

residuals=y_test-results["Random Forest"]['Predicted']

plt.figure(figsize=(6,4))
sns.kdeplot(residuals,fill=True)
plt.title("Residual Error Distribution (Random Forest)")
plt.xlabel("Prediction Error")
plt.tight_layout()

residual_chart=os.path.join(charts_dir,"residual_distribution.png")
plt.savefig(residual_chart,dpi=300,bbox_inches="tight")
plt.close()

rf=models["Random Forest"]

importance=rf.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(features,importance)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=30)
plt.tight_layout()

fi_chart=os.path.join(charts_dir,"feature_importance.png")
plt.savefig(fi_chart,dpi=300,bbox_inches="tight")
plt.close()

prediction_table=pd.DataFrame()

for model in results:
    temp=results[model].copy()
    temp.columns=[model+"_Actual",model+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"traffic_predictions.csv")
prediction_table.to_csv(csv_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Traffic Volume Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="""This report presents a machine learning approach for predicting traffic volume using regression models. 
The models analyze temporal and environmental variables such as hour of day, weekday patterns, weather, and event indicators to forecast traffic flow. 
Multiple regression algorithms were evaluated including Linear Regression, Ridge Regression, Support Vector Regression, and Random Forest."""

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():

    line=f"{row['Model']} | MAE: {row['MAE']:.2f} | RMSE: {row['RMSE']:.2f} | R2 Score: {row['R2']:.2f}"

    pdf.cell(0,8,line,0,1)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"RMSE Model Comparison",0,1)
pdf.image(rmse_chart,w=170)

pdf.ln(5)
pdf.multi_cell(0,8,"This chart compares the prediction error of each regression model using RMSE. Lower RMSE values indicate more accurate predictions.")

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Residual Error Distribution",0,1)
pdf.image(residual_chart,w=170)

pdf.ln(5)
pdf.multi_cell(0,8,"Residual distribution illustrates the spread of prediction errors. A tighter distribution indicates better model reliability.")

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Importance (Random Forest)",0,1)
pdf.image(fi_chart,w=150)

pdf.ln(5)
pdf.multi_cell(0,8,"Feature importance analysis reveals which variables most influence traffic predictions. Time-of-day and event indicators are typically dominant predictors.")

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Prediction Table Samples",0,1)

for model in results:

    pdf.ln(3)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,model+" Predictions",0,1)

    sample=results[model].head(10)

    pdf.set_font("Arial","B",11)
    pdf.cell(45,8,"Actual Volume",1)
    pdf.cell(45,8,"Predicted Volume",1)
    pdf.ln()

    pdf.set_font("Arial","",11)

    for i,row in sample.iterrows():
        pdf.cell(45,8,str(round(row['Actual'],2)),1)
        pdf.cell(45,8,str(round(row['Predicted'],2)),1)
        pdf.ln()

    pdf.ln(6)

pdf_path=os.path.join(output_dir,"Traffic_Prediction_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)