"""

Para rodar no localhost:
Pelo terminal inserir: uvicorn <nome do arquivo .py>:<nome da aplicação> --reload
Ex: uvicorn FastAPI:app --reload

"""


import sqlite3
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


#carregar dados
con = sqlite3.connect('dados-treino.db')
df_dim = pd.read_sql_query(f'SELECT * FROM df_dim', con)
classificar_contas = pd.read_sql_query(f'SELECT * FROM classificar_contas', con)

#carregar modelo
ml = joblib.load('regressaologistica.joblib')


app = FastAPI()


class DataAccounts(BaseModel):
    account_number: int
    occupation: str
    city: str
    state: str
    age: int     
        
dados_contas = []
contas_nao_classificadas = df_dim[df_dim['is_Fraud'].isin([np.NaN])]

for row in range(0, len(contas_nao_classificadas)):
    dados_contas.append(
        DataAccounts(
            account_number = contas_nao_classificadas.iloc[row]['account_number'],
            occupation = contas_nao_classificadas.iloc[row]['occupation'],
            city = contas_nao_classificadas.iloc[row]['city'],
            state = contas_nao_classificadas.iloc[row]['state'],
            age = contas_nao_classificadas.iloc[row]['age'],
            is_Fraud = contas_nao_classificadas.iloc[row]['is_Fraud'],
        )
    )
    

@app.get("/")
def instruction():
    """
    **URL/accounts**: <br>
    Disponibiliza todas as contas ainda não classificadas como lícitas ou ilícitas do banco de dados. 
    
    **URL/predict**: <br>
    Diponibiliza um endpoint GET que recebe como parâmetro na URL o número da conta e retorna sua classificação
    como lícita ou ilícita. A rota segue a seguinte estrutura URL/predict/?account_number=12345 
    """
    return {'instrução': 'Acesse ../docs para visualizar funcionalidades da API'}
    
    
@app.get("/accounts")
def accounts():
    """
    **Disponibiliza todas as contas ainda não classificadas como lícitas ou ilícitas do banco de dados**.
    """
    return dados_contas


@app.get("/predict")
def predict(account_number: int):
    """
    **Classifica como lícitas ou ilícitas as contas ainda não classificadas do banco de dados.**
    
    **Entrada**: <br>
    account_number -> número da conta
    
    **Saída**: <br>
    account_number -> número da conta <br>
    is_Fraud -> 0 para conta lícita e 1 para ilícita <br>
    probability_Fraud -> probabilidade da conta ser ilícita <br>
    
    Número de conta não identificada -> caso o número da conta não esteja disponível para classificação no banco de dados
    """
    try:
        indice = df_dim[df_dim['account_number'] == account_number].index[0]
        classificar = classificar_contas[classificar_contas['index'] == indice].set_index('index')
        pred = ml.predict(classificar)
        pred_prob = ml.predict_proba(classificar)
        return {'account_number': account_number, 'is_Fraud': int(pred[0]), 'probability_Fraud': round(pred_prob[0][1], 4)}
    
    except:
        return 'Número de conta não identificada.'
    