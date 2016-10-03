from flask import render_template, request
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from app import app
#from ModelIt import ModelIt

# user = 'moon' #add your username here (same as previous postgreSQL)            
# host = 'localhost'
# dbname = 'hhc_hos'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(databa90.0se = dbname, user = user)

@app.route('/')
def hhc_input():
    #enter ccn on input page
    return render_template("input.html")
    
@app.route('/output')
def hhc_output():
  #pull the query of ccn from database
  #save the query on the output page and predict
  ccn = request.args.get('ccn')
  print('ccn:', ccn, type(ccn))
  ccn = int(ccn)
  data = pd.read_csv('/home/ubuntu/insight_home_care_data/hhc_2016/worse_and_same_hhc.csv',
                     index_col = 'ccn')
    #read 'ccn' as the index!
  query = data.loc[ccn, :]
  if query['er_cd'] == 1:
    output = 'good_output.html'
  else:
    output = 'output.html'

  #predict probability
  simu_group = pd.read_csv('/home/ubuntu/insight_home_care_data/hhc_2016/simu_group_2class.csv', 
                             index_col = 'ccn')
  x_df = pd.concat([simu_group.ix[:, 'certify_yrs':'bedsore_ck'], 
                    simu_group.ix[:,'othr_cnt':'female_65']], axis=1)
  x = np.array(x_df)
  y = np.array(simu_group['er_cd'])   
  x_query = x_df.loc[ccn, :]
  #if duplicate queries, use only one
  if len(x_query.shape) > 1:
      x_query = x_query.iloc[0, :]

  rf_grid = RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_features = 'sqrt', class_weight = 'balanced', random_state = 7)
  rf_grid.fit(x, y) 
  class_proba = rf_grid.predict_proba(x_query)

  return render_template(output, query=query, ccn=ccn, class_proba=class_proba[0])

@app.route('/predict')
def hhc_predict():
    #get new predictors from output page
    ccn = request.args.get('ccn')
    rn_cnt =  request.args.get('rn_cnt')
    print('ccn:', ccn, type(ccn))
    print('rn_cnt:', rn_cnt, type(rn_cnt))
    pneu_shot =  request.args.get('pneu_shot')
    flu_shot =  request.args.get('flu_shot')
    timely =  request.args.get('timely')
    phys_cnt =  request.args.get('phys_cnt')
    drug_edu =  request.args.get('drug_edu')  
    depr_ck = request.args.get('depr_ck') 
    ccn = int(ccn)
    rn_cnt = float(rn_cnt)
    pneu_shot = float(pneu_shot)
    flu_shot = float(flu_shot)
    timely = float(timely)
    phys_cnt = float(phys_cnt)
    drug_edu = float(drug_edu)
    depr_ck = float(depr_ck)
    
    data = pd.read_csv('/home/ubuntu/insight_home_care_data/hhc_2016/worse_and_same_hhc.csv',
                     index_col = 'ccn')
    #read 'ccn' as the index!
    query = data.loc[ccn, :]
    
    #pull the query of ccn and change to new predictors
    simu_group = pd.read_csv('/home/ubuntu/insight_home_care_data/hhc_2016/simu_group_2class.csv', 
                             index_col = 'ccn')
    x_df = pd.concat([simu_group.ix[:, 'certify_yrs':'bedsore_ck'], 
                      simu_group.ix[:,'othr_cnt':'female_65']], axis=1)
    x = np.array(x_df)
    y = np.array(simu_group['er_cd'])   
    x_query = x_df.loc[ccn, :]
    #if duplicate queries, use only one
    if len(x_query.shape) > 1:
        x_query = x_query.iloc[0, :]
    x_query['rn_cnt'], x_query['pneu_shot'], x_query['flu_shot'], x_query['depr_ck'] = rn_cnt, pneu_shot, flu_shot, depr_ck
    x_query['timely'], x_query['phys_cnt'], x_query['drug_edu'] = timely, phys_cnt, drug_edu
    
    #run model and predict new probability
    rf_grid = RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_features = 'sqrt', class_weight = 'balanced', random_state = 7)
    rf_grid.fit(x, y) 
    class_proba = rf_grid.predict_proba(x_query)

    #svm = SVC(probability=True)
    #svm.fit(x, y)
    #class_proba = svm.predict_proba(x_query)
    
    """
    #plot the probability on the pie chart
    fig = plt.figure(figsize=(6.67, 4.67))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.0, bottom=0.0, top=1, right=1)
    patches, test, autotext = ax.pie(class_proba[0], 
                                     autopct='%1.1f%%', explode=[0.0, 0], 
                                     startangle=90,
                                     colors = ['lightskyblue', 'yellowgreen'])
    autotext[0].set_fontsize = 30.0
    autotext[1].set_fontsize = 30.0
    fig.legend(patches, ['Unsatisfied', 'Satisfied'], loc="best")
    """
    #matplotlib.style.use('ggplot')
    fig = plt.figure(figsize=(6.67, 4.67))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.25)
    data = pd.DataFrame(class_proba, index=['Probability'], columns=['Bad', 'Good'])
    data.plot(kind='barh', stacked=True, width=0.1, ax=ax)
    tmp_bar = tempfile.NamedTemporaryFile(dir='/home/ubuntu/insight2016_Boston/app/static/tmp',
                                    suffix='.png',delete=False)
    fig.savefig(tmp_bar, dpi=150)
    tmp_bar.close()
    bar_proba = tmp_bar.name.split('/')[-1] 
    #print(pie_proba)

    return render_template("predict.html", ccn=ccn,
                           x_query=x_query, query=query, class_proba=class_proba[0],
                           bar_proba=bar_proba)

    

