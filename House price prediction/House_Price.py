import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
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
'LotArea':np.random.randint(5000,15000,300),
'OverallQual':np.random.randint(3,10,300),
'GrLivArea':np.random.randint(800,3500,300),
'YearBuilt':np.random.randint(1970,2023,300),
'FullBath':np.random.randint(1,4,300),
'HalfBath':np.random.randint(0,2,300),
'TotRmsAbvGrd':np.random.randint(4,10,300),
'SalePrice':np.random.randint(100000,500000,300)
})

features=['LotArea','OverallQual','GrLivArea','YearBuilt','FullBath','HalfBath','TotRmsAbvGrd']
target='SalePrice'

scaler=StandardScaler()

X=data[features]
X_scaled=scaler.fit_transform(X)

y=data[target]

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
'Linear Regression':LinearRegression(),
'Ridge Regression':Ridge(),
'Lasso Regression':Lasso(),
'Random Forest':RandomForestRegressor(n_estimators=200,random_state=42)
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
        'Predicted':np.round(pred,2)
    })

    results[name]=result

    plt.figure(figsize=(8,4))
    plt.scatter(result['Actual'],result['Predicted'])
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(name+" Prediction")
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,name.replace(" ","_")+"_chart.png")
    plt.savefig(chart_path)
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Model','MAE','RMSE','R2'])

prediction_table=pd.DataFrame()

for model in results:
    temp=results[model].copy()
    temp.columns=[model+"_Actual",model+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"prediction_results.csv")
prediction_table.to_csv(csv_path,index=False)

metrics_path=os.path.join(output_dir,"model_metrics.csv")
metrics_df.to_csv(metrics_path,index=False)

rf=models['Random Forest']
rf.fit(X_train,y_train)

importance=rf.feature_importances_

imp_df=pd.DataFrame({
'Feature':features,
'Importance':importance
}).sort_values(by='Importance',ascending=False)

plt.figure(figsize=(8,4))
plt.bar(imp_df['Feature'],imp_df['Importance'])
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.tight_layout()

feature_chart=os.path.join(charts_dir,"feature_importance.png")
plt.savefig(feature_chart)
plt.close()

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"House Price Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="""This report presents a machine learning based house price prediction system using regression models. The system predicts property prices using structural attributes such as lot area, living area, quality rating and number of rooms. Multiple regression algorithms were evaluated to determine the most accurate predictive model."""

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():

    line=f"{row['Model']} | MAE: {row['MAE']:.2f} | RMSE: {row['RMSE']:.2f} | R2 Score: {row['R2']:.2f}"

    pdf.cell(0,8,line,0,1)

pdf.add_page()

for model in models:

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,model+" Prediction Chart",0,1)

    img=os.path.join(charts_dir,model.replace(" ","_")+"_chart.png")
    pdf.image(img,w=170)

    pdf.ln(5)

    explanation=f"This visualization compares actual house prices with the predicted values produced by the {model} model. The closer the points align to the diagonal trend, the more accurate the predictions."

    pdf.set_font("Arial","",11)
    pdf.multi_cell(0,8,explanation)

    pdf.ln(5)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Importance (Random Forest)",0,1)

pdf.image(feature_chart,w=170)

pdf.ln(5)

pdf.set_font("Arial","",11)

text2="""The feature importance chart illustrates which property attributes most strongly influence the predicted house price. Factors such as overall quality and living area typically have the highest impact on price estimation."""

pdf.multi_cell(0,8,text2)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Prediction Table Sample",0,1)

sample=prediction_table.head(10)

pdf.set_font("Arial","",9)

for col in sample.columns:
    pdf.cell(40,8,col,1)

pdf.ln()

for i,row in sample.iterrows():
    for val in row:
        pdf.cell(40,8,str(round(val,2)),1)
    pdf.ln()

pdf_path=os.path.join(output_dir,"House_Price_Prediction_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)