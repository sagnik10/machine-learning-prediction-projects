import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score
from fpdf import FPDF
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'ApplicantIncome':np.random.randint(2000,15000,400),
'CoapplicantIncome':np.random.randint(0,8000,400),
'LoanAmount':np.random.randint(50,500,400),
'Loan_Amount_Term':np.random.choice([180,240,300,360],400),
'Credit_History':np.random.choice([0,1],400),
'Dependents':np.random.randint(0,4,400),
'Education':np.random.choice([0,1],400),
'Self_Employed':np.random.choice([0,1],400)
})

data['Loan_Status']=np.where(
(data['ApplicantIncome']+data['CoapplicantIncome']>6000) &
(data['Credit_History']==1),1,0)

features=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Dependents','Education','Self_Employed']
target='Loan_Status'

X=data[features]
y=data[target]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

models={
'Logistic Regression':LogisticRegression(),
'Decision Tree':DecisionTreeClassifier(),
'Random Forest':RandomForestClassifier(n_estimators=200,random_state=42),
'SVM':SVC(probability=True),
'Gradient Boosting':GradientBoostingClassifier()
}

metrics=[]
results={}

for name,model in models.items():

    model.fit(X_train,y_train)

    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]

    acc=accuracy_score(y_test,pred)
    prec=precision_score(y_test,pred)
    rec=recall_score(y_test,pred)
    f1=f1_score(y_test,pred)
    roc=roc_auc_score(y_test,prob)

    metrics.append([name,acc,prec,rec,f1,roc])

    result=pd.DataFrame({
        'Actual':y_test.values,
        'Predicted':pred
    })

    results[name]=result

    plt.figure(figsize=(6,4))
    plt.scatter(range(len(result)),result['Actual'],label='Actual')
    plt.scatter(range(len(result)),result['Predicted'],label='Predicted')
    plt.title(name+" Prediction")
    plt.legend()
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,name.replace(" ","_")+"_prediction.png")
    plt.savefig(chart_path)
    plt.close()

    cm=confusion_matrix(y_test,pred)

    fig,ax=plt.subplots(figsize=(6,5))

    im=ax.imshow(cm,cmap='Blues')

    ax.set_title(name+" Confusion Matrix")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["Rejected","Approved"])
    ax.set_yticklabels(["Rejected","Approved"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="black",fontsize=12)

    fig.colorbar(im)

    plt.tight_layout()

    cm_path=os.path.join(charts_dir,name.replace(" ","_")+"_confusion.png")
    plt.savefig(cm_path,dpi=300)
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Model','Accuracy','Precision','Recall','F1','ROC_AUC'])

prediction_table=pd.DataFrame()

for model in results:
    temp=results[model].copy()
    temp.columns=[model+"_Actual",model+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"loan_predictions.csv")
prediction_table.to_csv(csv_path,index=False)

metrics_path=os.path.join(output_dir,"model_metrics.csv")
metrics_df.to_csv(metrics_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Loan Approval Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="""This report presents a machine learning system developed to predict loan approval decisions based on applicant financial and demographic attributes. Multiple classification models including Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, and Gradient Boosting were evaluated. The models analyze features such as income, loan amount, credit history, employment status, and dependents to determine approval probability."""

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Model Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():

    line=f"{row['Model']} | Accuracy: {row['Accuracy']:.2f} | Precision: {row['Precision']:.2f} | Recall: {row['Recall']:.2f} | F1: {row['F1']:.2f} | ROC-AUC: {row['ROC_AUC']:.2f}"

    pdf.cell(0,8,line,0,1)

pdf.add_page()

for model in models:

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,model+" Prediction Chart",0,1)

    img=os.path.join(charts_dir,model.replace(" ","_")+"_prediction.png")
    pdf.image(img,w=170)

    pdf.ln(5)

    explanation=f"This visualization compares predicted and actual loan approval outcomes generated by the {model} model. It demonstrates how well the model identifies eligible applicants based on financial indicators."

    pdf.set_font("Arial","",11)
    pdf.multi_cell(0,8,explanation)

    pdf.ln(5)

    pdf.cell(0,10,model+" Confusion Matrix",0,1)

    cm_img=os.path.join(charts_dir,model.replace(" ","_")+"_confusion.png")
    pdf.image(cm_img,w=120)

    pdf.ln(5)

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
        pdf.cell(40,8,str(val),1)
    pdf.ln()

pdf_path=os.path.join(output_dir,"Loan_Approval_Prediction_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)