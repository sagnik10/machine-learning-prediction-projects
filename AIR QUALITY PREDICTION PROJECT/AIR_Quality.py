import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fpdf import FPDF
import os

base_dir=os.path.dirname(os.path.abspath(__file__))
output_dir=os.path.join(base_dir,"output")
charts_dir=os.path.join(output_dir,"charts")

os.makedirs(charts_dir,exist_ok=True)

np.random.seed(42)

data=pd.DataFrame({
'Date':pd.date_range(start='2024-01-01',periods=200,freq='D'),
'Temperature':np.random.randint(25,38,200),
'Humidity':np.random.randint(30,70,200),
'WindSpeed':np.random.randint(2,15,200),
'PM2.5':np.random.randint(50,150,200),
'PM10':np.random.randint(80,200,200),
'NO2':np.random.randint(20,80,200),
'CO':np.round(np.random.uniform(0.5,2.5,200),2),
'SO2':np.random.randint(5,30,200),
'O3':np.random.randint(10,60,200)
})

features=['Temperature','Humidity','WindSpeed']
targets=['PM2.5','PM10','NO2','CO','SO2','O3']

scaler=StandardScaler()

X=data[features]
X_scaled=scaler.fit_transform(X)

results={}
metrics=[]

for target in targets:

    y=data[target]

    X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

    model=LinearRegression()
    model.fit(X_train,y_train)

    pred=model.predict(X_test)

    mae=mean_absolute_error(y_test,pred)
    mse=mean_squared_error(y_test,pred)
    r2=r2_score(y_test,pred)

    metrics.append([target,mae,mse,r2])

    result=pd.DataFrame({
        'Actual':y_test.values,
        'Predicted':np.round(pred,2)
    })

    results[target]=result

    plt.figure(figsize=(8,4))
    plt.plot(result['Actual'].values,marker='o',label='Actual')
    plt.plot(result['Predicted'].values,marker='x',label='Predicted')
    plt.title(target+" Prediction")
    plt.xlabel("Samples")
    plt.ylabel(target)
    plt.legend()
    plt.tight_layout()

    chart_path=os.path.join(charts_dir,target+"_chart.png")
    plt.savefig(chart_path)
    plt.close()

metrics_df=pd.DataFrame(metrics,columns=['Pollutant','MAE','MSE','R2'])

prediction_table=pd.DataFrame()

for pollutant in results:
    temp=results[pollutant].copy()
    temp.columns=[pollutant+"_Actual",pollutant+"_Predicted"]
    prediction_table=pd.concat([prediction_table,temp],axis=1)

csv_path=os.path.join(output_dir,"prediction_results.csv")
prediction_table.to_csv(csv_path,index=False)

class PDF(FPDF):
    pass

pdf=PDF()
pdf.set_auto_page_break(auto=True,margin=15)
pdf.add_page()

pdf.set_font("Arial","B",18)
pdf.cell(0,10,"Air Quality Prediction Report",0,1,'C')

pdf.set_font("Arial","",12)

text="""This report presents results of a machine learning based air quality prediction system using Linear Regression. The model predicts pollutant concentrations using environmental parameters including temperature, humidity and wind speed. Charts below illustrate actual vs predicted pollutant levels."""

pdf.multi_cell(0,8,text)

pdf.ln(5)

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Evaluation Metrics",0,1)

pdf.set_font("Arial","",11)

for i,row in metrics_df.iterrows():
    line=f"{row['Pollutant']} | MAE: {row['MAE']:.2f} | MSE: {row['MSE']:.2f} | R2 Score: {row['R2']:.2f}"
    pdf.cell(0,8,line,0,1)

pdf.add_page()

for pollutant in targets:

    pdf.set_font("Arial","B",14)
    pdf.cell(0,10,pollutant+" Prediction Chart",0,1)

    img=os.path.join(charts_dir,pollutant+"_chart.png")
    pdf.image(img,w=170)

    pdf.ln(5)

    explanation=f"The chart compares actual and predicted {pollutant} levels. The regression model estimates pollutant concentration based on meteorological conditions and captures general pollution trends."

    pdf.set_font("Arial","",11)
    pdf.multi_cell(0,8,explanation)

    pdf.ln(5)

pdf.add_page()

pdf.set_font("Arial","B",14)
pdf.cell(0,10,"Prediction Table Sample",0,1)

sample=prediction_table.head(10)

pdf.set_font("Arial","",9)

for col in sample.columns:
    pdf.cell(30,8,col,1)

pdf.ln()

for i,row in sample.iterrows():
    for val in row:
        pdf.cell(30,8,str(round(val,2)),1)
    pdf.ln()

pdf_path=os.path.join(output_dir,"Air_Quality_Prediction_Report.pdf")
pdf.output(pdf_path)

print(metrics_df)
print("\nOutputs saved inside:",output_dir)