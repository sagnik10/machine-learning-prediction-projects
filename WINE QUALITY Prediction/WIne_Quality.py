import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from fpdf import FPDF
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'fixed_acidity':np.random.uniform(6,9,400),
'volatile_acidity':np.random.uniform(0.1,1,400),
'citric_acid':np.random.uniform(0,1,400),
'residual_sugar':np.random.uniform(1,10,400),
'chlorides':np.random.uniform(0.03,0.1,400),
'free_sulfur_dioxide':np.random.randint(10,70,400),
'total_sulfur_dioxide':np.random.randint(30,200,400),
'density':np.random.uniform(0.990,1.005,400),
'pH':np.random.uniform(3,4,400),
'sulphates':np.random.uniform(0.3,1,400),
'alcohol':np.random.uniform(8,14,400)
})

data['quality']=(data['alcohol']*0.6-data['volatile_acidity']*2+np.random.normal(0,1,400)).round()

data['quality_label']=data['quality'].apply(lambda x:1 if x>=6 else 0)

X=data.drop(['quality','quality_label'],axis=1)
y=data['quality_label']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
"Logistic Regression":LogisticRegression(max_iter=300),
"Random Forest":RandomForestClassifier(n_estimators=200,random_state=42),
"XGBoost":XGBClassifier(eval_metric='logloss',random_state=42),
"ANN":MLPClassifier(hidden_layer_sizes=(100,100),max_iter=500)
}

metrics=[]
results={}

for name,model in models.items():

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))

    metrics.append([name,acc,rmse])

    result=pd.DataFrame({
        "Actual":y_test.values,
        "Predicted":pred
    })

    results[name]=result

    plt.figure(figsize=(6,4))
    plt.scatter(result['Actual'],result['Predicted'],alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(name+" Prediction Scatter")
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,name.replace(" ","_")+"_scatter.png")
    plt.savefig(chart_path,dpi=300)
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Model','Accuracy','RMSE'])

plt.figure(figsize=(7,4))
plt.bar(metrics_df['Model'],metrics_df['Accuracy'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()

accuracy_chart=os.path.join(charts_dir,"model_accuracy.png")
plt.savefig(accuracy_chart,dpi=300)
plt.close()

corr=data.corr()

plt.figure(figsize=(10,7))
sns.heatmap(corr,cmap="coolwarm",linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

heatmap_chart=os.path.join(charts_dir,"correlation_heatmap.png")
plt.savefig(heatmap_chart,dpi=300)
plt.close()

rf=models["Random Forest"]
importance=rf.feature_importances_

plt.figure(figsize=(7,4))
plt.barh(X.columns,importance)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()

fi_chart=os.path.join(charts_dir,"feature_importance.png")
plt.savefig(fi_chart,dpi=300)
plt.close()

cm=confusion_matrix(y_test,results["ANN"]["Predicted"])

plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Low","High"],yticklabels=["Low","High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - ANN")
plt.tight_layout()

cm_chart=os.path.join(charts_dir,"confusion_matrix.png")
plt.savefig(cm_chart,dpi=300)
plt.close()

prediction_table=pd.DataFrame()

for model in results:
    temp=results[model].copy()
    temp.columns=[model+"_Actual",model+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"wine_predictions.csv")
prediction_table.to_csv(csv_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)

pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Wine Quality Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="Machine learning models were trained to predict wine quality using physicochemical properties such as acidity, sulphates and alcohol."

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():
    line=f"{row['Model']} | Accuracy: {row['Accuracy']:.2f} | RMSE: {row['RMSE']:.2f}"
    pdf.cell(0,8,line,0,1)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Accuracy Comparison",0,1)
pdf.image(accuracy_chart,w=170)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Correlation Heatmap",0,1)
pdf.image(heatmap_chart,w=170)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Feature Importance",0,1)
pdf.image(fi_chart,w=150)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Confusion Matrix (ANN)",0,1)
pdf.image(cm_chart,w=120)

pdf_path=os.path.join(output_dir,"Wine_Quality_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)