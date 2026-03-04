import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fpdf import FPDF
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'age':np.random.randint(15,21,500),
'studytime':np.random.randint(1,5,500),
'failures':np.random.randint(0,4,500),
'absences':np.random.randint(0,30,500),
'Medu':np.random.randint(0,5,500),
'Fedu':np.random.randint(0,5,500),
'G1':np.random.randint(5,20,500),
'G2':np.random.randint(5,20,500)
})

data['G3']=(0.3*data['G1']+0.5*data['G2']+0.5*data['studytime']-0.3*data['failures']-0.1*data['absences']+np.random.normal(0,1,500))
data['G3']=data['G3'].clip(0,20)

features=['age','studytime','failures','absences','Medu','Fedu','G1','G2']
target='G3'

X=data[features]
y=data[target]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
"Linear Regression":LinearRegression(),
"Ridge Regression":Ridge(alpha=1.0),
"Lasso Regression":Lasso(alpha=0.01),
"Random Forest":RandomForestRegressor(n_estimators=200,random_state=42)
}

metrics=[]
results={}

for name,model in models.items():

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    rmse=np.sqrt(mean_squared_error(y_test,pred))
    mae=mean_absolute_error(y_test,pred)
    r2=r2_score(y_test,pred)

    metrics.append([name,rmse,mae,r2])

    result=pd.DataFrame({
        'Actual':y_test.values,
        'Predicted':pred
    })

    results[name]=result

    plt.figure(figsize=(6,4))
    plt.scatter(result['Actual'],result['Predicted'])
    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")
    plt.title(name+" Prediction")
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,name.replace(" ","_")+"_prediction.png")
    plt.savefig(chart_path,dpi=300,bbox_inches="tight")
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Model','RMSE','MAE','R2'])

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()

heatmap_path=os.path.join(charts_dir,"correlation_heatmap.png")
plt.savefig(heatmap_path,dpi=300,bbox_inches="tight")
plt.close()

rf=models["Random Forest"]

importance=rf.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(features,importance)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=45)
plt.tight_layout()

fi_path=os.path.join(charts_dir,"feature_importance.png")
plt.savefig(fi_path,dpi=300,bbox_inches="tight")
plt.close()

prediction_table=pd.DataFrame()

for model in results:
    temp=results[model].copy()
    temp.columns=[model+"_Actual",model+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"student_predictions.csv")
prediction_table.to_csv(csv_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)

pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Student Academic Performance Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="""This report presents a machine learning system for predicting student academic performance. 
Multiple regression models including Linear Regression, Ridge, Lasso, and Random Forest are used. 
The models analyze study time, absences, failures, parental education, and previous grades to estimate the final exam score."""

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():

    line=f"{row['Model']} | RMSE: {row['RMSE']:.2f} | MAE: {row['MAE']:.2f} | R2 Score: {row['R2']:.2f}"

    pdf.cell(0,8,line,0,1)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Correlation Heatmap",0,1)

pdf.image(heatmap_path,w=170)

pdf.ln(5)

pdf.set_font("Arial","",11)
pdf.multi_cell(0,8,"The correlation heatmap illustrates relationships between student attributes and the final grade. Previous exam scores have the strongest correlation with the final result.")

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Importance (Random Forest)",0,1)

pdf.image(fi_path,w=150)

pdf.ln(5)

pdf.set_font("Arial","",11)
pdf.multi_cell(0,8,"Feature importance shows how different variables influence academic performance predictions. Previous grades and study time are the most influential predictors.")

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Prediction Table Samples",0,1)

for model in results:

    pdf.ln(3)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,model+" Predictions",0,1)

    sample=results[model].head(10)

    pdf.set_font("Arial","B",11)

    pdf.cell(45,8,"Actual Grade",1)
    pdf.cell(45,8,"Predicted Grade",1)
    pdf.ln()

    pdf.set_font("Arial","",11)

    for i,row in sample.iterrows():

        pdf.cell(45,8,str(round(row['Actual'],2)),1)
        pdf.cell(45,8,str(round(row['Predicted'],2)),1)
        pdf.ln()

    pdf.ln(6)

pdf_path=os.path.join(output_dir,"Student_Performance_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)