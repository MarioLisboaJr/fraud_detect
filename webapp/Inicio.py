#bibliotecas
import streamlit as st


#configuração padrão da página
st.set_page_config(page_title='Fraud Detect', layout='wide')

#descrição
st.markdown('<div style="text-align: center"><h1> EVITANDO FRAUDES BANCÁRIAS </div>',unsafe_allow_html=True)

st.caption('<div style="text-align: center"><h2> CLASSIFICAÇÃO DE CONTAS ILÍCITAS <br><br><br></div>',unsafe_allow_html=True)

st.image('https://static.wixstatic.com/media/11062b_6ed150d7b33842c5938db540a502ea15~mv2.jpg/v1/fill/w_1175,h_550,al_c,q_85,usm_0.66_1.00_0.01/Internet%20Banking.webp')

md = '''
Uma instituição financeira de pagamentos classifica seus clientes de acordo com a forma que a conta é utilizada em sua plataforma. Contas classificadas como suspeitas de fraude são imediatamente encerradas. Deseja-se então criar um modelo de machine learning afim de detectar essas contas ilíctas e minimizar os problemas decorrentes destas atividades.
'''
st.write(f'<div style="text-align: justify"> <br><br> {md} </div>',unsafe_allow_html=True)

st.caption('<div style="text-align: left"><h2> OBJETIVO </div>',unsafe_allow_html=True)

md = '''

**Identificar contas ilícitas em uma instituição financeira de pagamentos.**

Desenvolvimento: <br>

- Conectar a database do SQLite <br>
- Realizar análise descritiva dos dados <br>
- Desenvolver um modelo de Machine Learning para Classificação <br>
- Classificar as contas <br>
- Disponibilizar a classificação para consulta em um arquivo csv <br>
- Criar uma RestAPI capaz de classificar as contas ainda não classificadas do banco de dados <br>

<br>

**Acesse a RestAPI**: https://api-fraud-detect.herokuapp.com/docs
'''
st.write(f'<div style="text-align: left"> {md} </div>',unsafe_allow_html=True)

st.caption('<div style="text-align: left"><h2> INFORMAÇÃO SOBRE OS DADOS </div>',unsafe_allow_html=True)

md = '''

**Os dados destes problema estão divididos em seis tabelas de um banco SQLite estruturado da seguinte maneira:** <br>

1. **accounts** <br>
Tabela que apresenta as informações cadastrais de cada conta. <br>
**id**: Identificador da tabela <br>
**account_number**: Número da conta <br>
**birth**: Data de nascimento <br>
**occupation**: Tipo de negócio autodeclarado <br>
**email**: E-mail da conta <br>
**address_id**: Identificador da tabela address <br>
**created_at**: Data de criação da conta <br>

2. **address** <br>
Tabela que identifica os pares de estado e cidade. <br>
**id**: Identificador da tabela <br>
**state**: Estado do cliente <br>
**city**: Cidade do cliente <br>
**created_at**: Data de cadastro da cidade <br>

3. **levels** <br>
Cada conta recebe uma classificação de acordo com a forma que utiliza a plataforma. Contas que utilizam com maior consistência ou com grande potêncial podem receber uma melhor classificação (A>B>C>D). Caso identifique-se que a conta possui características suspeitas, de fraude, é atribuida a categoria F e executado o encerramento. <br>
**id**: Identificador da tabela <br>
**account_number**: Número da conta <br>
**level**: A, B, C, D e F <br>
**created_at**: Data da classificação <br>

4. **charges** <br>
Tabela apresenta as emissões de boletos realizadas pelos clientes com os respectivos status de pago ou não. <br>
**id**: Identificador da tabela <br>
**account_number**: Número da conta <br>
**status**: Status da cobrança (paid, unpaid) <br>
**value**: Valor da cobrança (em centavos) <br>
**created_at**: Data de criação do boleto <br>

5. **transaction_type** <br>
Tabela que permite identificar qual o tipo de cada transação da tabela transactions. <br>
**id**: Identificador da tabela <br>
**description**: 'boleto_recebido', 'pix_enviado', 'pix_recebido' <br>
**description_long**: 'BOLETO RECEBIDO PELO CLIENTE', 'PIX ENVIADO PELO CLIENTE PARA UMA CONTA EXTERNA', 'PIX RECEBIDO PELO CLIENTE' <br>

6. **transactions** <br>
Tabela com as transações efetivadas por cada conta, logo, caso um boleto tenha sido pago esta informação estará presente nesta tabela e na tabela charges. <br>
**id**: Identificador da tabela <br>
**account_number**: Número da conta <br>
**transaction_type_id**: Identificador da tabela transaction_type <br>
**value**: Valor da transação (em centavos) <br>
**created_at**: Data da transação <br>
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)

st.caption('<div style="text-align: left"><h2> ANÁLISE EXPLORATÓRIA </div>',unsafe_allow_html=True)

md='''

**Insights obtidos dos dados:**

- Existe uma tendência maior das contas ilícitas estarem localizadas nos estados do PR e RJ;
- Aparentemente o estado de MG é o mais confiável;
- Devemos ter mais atenção também quando a pessoa autodeclara sua ocupação como 'Outros'. Existe um correlacionamento maior dessas pessoas a atividades suspeitas e, coincidentemente ou não, elas estão mais localizadas no PR;
- Analistas e Desenvolvedores tendem a ter menos suas contas suspendidas;
- A relação entre a idade e as contas classificadas com F é inversa, ou seja, contas ilícitas estão mais relacionadas a pessoas mais jovens;
- As contas da categoria B tendem a receber mais, indo de encontro ao esperado que era a categoria A. Ao que parece, a vantagem da categoria A é que ela é mais uniforme entre valores já recebidos e valores futuros a receber. Assim, além de serem a segunda melhor correlacionada aos recebimentos, são também mais previsíveis, podendo as tornar mais vantajosas;
- As contas classificadas como F tendem a ter um valor médio dos pix enviados menor;
- Em contas lícitas, os valores médios recebidos por pix e boletos e, valores médios dos boletos a receber, tendem a ser maiores.

**Abaixo podemos visualizar melhor estas informações:**
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)

st.image('https://raw.githubusercontent.com/MarioLisboaJr/fraud_detect/main/outputs/output_30_0.png')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_31_0.png?raw=true')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_31_1.png?raw=true')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_32_0.png?raw=true')

md='''
Sobre a relação da idade com as contas ilícitas, abaixo podemos observar que, enquanto nosso público estudado possui idades entre 18 e 67 anos, as contas ilícitas foram identificadas num grupo menor que possui entre 22 e 47 anos, sendo a grande maioria entre 29 e 39 anos.
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)

st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_37_0.png?raw=true')

md='''
Para entender melhor a diferença entre as movimentações financeiras de clientes com contas lícitas e ilícitas, abaixo temos a visualização gráfica através de boxplots e fica possível observar que:

**Valores dos boletos (não recebidos/recebidos) e pix recebidos:**

- Nas contas classificadas como F a mediana dos valores fica próximo a 8.000 enquanto nas outras classes gira entorno de 5.000, 60% abaixo;
- Enquanto nas contas ilícitas os valores destes boletos chegam a quase 14.000, nas contas lícitas valores acima de 9.000 são considerados como exceções, 55% abaixo.

**Valores dos pix enviados:**

- Enquanto nos outros gráficos podemos reparar que os valores das transações nas contas classe F são maiores que nas demais, aqui já podemos perceber o contrário. Nas contas ilícitas os valores retirados atingem um máximo de 150.000, acima disso pode ser considerado como valores fora do padrão, com estes não chegando a 400.000. Já nas contas classe A, B, C e D, os valores retirados, normalmente, são de até 300.000. Podendo ultrapassar os 900.000. 
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)

st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_39_0.png?raw=true')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_39_1.png?raw=true')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_39_2.png?raw=true')
st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_39_3.png?raw=true')

st.caption('<div style="text-align: left"><h2> MACHINE LEARNING </div>',unsafe_allow_html=True)

md='''
Para criação do modelo foi testado cinco algoritmos diferentes: 

- Random Forest
- Suport Vector Machine
- Logistic Regression
- K Neighbor Nearest
- Gradient Booster

**Após o teste inicial o modelo Logistic Regression foi escolhido por algumas razões:**

**Existem duas situações importantes neste nosso problema:**

**1ª)**  A identificação errada de uma conta como ilícita, suspende a conta de um cliente injustamente. Isso gera um impacto negativo para a imagem da empresa e para a operação, que diminui sua captação. Para medir esta situação, onde o **custo dos Falso Positivo são altos, Precision é uma boa métrica de desempenho do modelo**;


**2ª)**  A não identificação das contas ilícitas gera para a empresa uma perda de credibilidade muito grande, podendo gerar uma associação entre a marca e fraudes bancárias. Assim, neste caso temos também um **alto custo dos Falsos Negativos, e uma métrica melhor para esta situação seria o Recall**.

Como temos de buscar atender as duas situações, analisar a ponderação entre Reccal e Precision em F1-Score, juntamente da Acucuracy do modelo, foi uma melhor ideia para avaliação dos resultados. Como Logistic Regression apresentou resulatados melhores nestes quesitos, foi eleito como melhor algoritmo para modelar nosso caso.
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)

st.caption('<div style="text-align: left"><h2> RESULTADOS DO MODELO </div>',unsafe_allow_html=True)

st.image('https://github.com/MarioLisboaJr/fraud_detect/blob/main/outputs/output_55_0.png?raw=true')

md='''
Relatório de Classificação (Logistic Regression):

               precision    recall  f1-score   support

         0.0       0.94      0.95      0.94        99
         1.0       0.87      0.85      0.86        40

    accuracy                           0.92       139

<br>
 
**Nosso modelo de Regressão Logística final leva em consideração sete características para predição das contas ilícitas.**

São elas:

**1)** Tipo de negócio autodeclarado pelo cliente; <br>
**2)** Código da cidade do cliente; <br>
**3)** Estado do cliente; <br>
**4)** Idade do cliente; <br>
**5)** Valor médio dos pix enviados pela conta; <br>
**6)** Valor médio dos boletos a receber da conta; <br>
**7)** Valor médio dos pix e boletos recebidos da conta. <br>

<br>

**O resultado obtido foi:**

- Identificação de 34 dos 40 clientes (85%) que possuíam contas classificadas como ilícitas;


- Acusação errada de 5 contas que não eram ilícitas como sendo. Este erro representa 5% dos clientes analisados.


- Modelo final atingiu uma precisão geral de 92% de acuracidade.

Um resultado consideravelmente bom.

<br>

**Das classificações erradas do modelo podemos analisar que:**

Das 5 contas classificadas como ilícitas pelo modelo mesmo não sendo:

- 3 eram classe D e 2 classe C;
- 3 contas foram elegidas como ilícitas com uma confiança de 66%, 67% e 53%. Sabendo que a classificação se dá em 50%, podemos notar que não existia muita certeza na classificação;
- 2 contas possuíam características fortes de contas ilícitas. Autodeclaração de ocupação era de 'Autonomo' e 'Outros' as duas mais correlacionadas com as contas suspeitas, ambas eram do RJ, segundo estado com mais índice de contas fraudulentas e possuíam 30 e 18 anos.

Das 6 contas classificadas como lícitas pelo modelo sendo da classe F:

- 3 contas não possuíam nenhuma transação;
- 3 contas foram eleitas como lícitas com probabilidade próxima 70%.

Pelos fatos descritos podemos também chegar à conclusão que além da classificação como lícita ou ilícita, a análise da probabilidade de classificação também pode ajudar a esclarecer possíveis equívocos de classificações incorretas. Um estudo mais aprofundado pode ajudar a melhorar a acuracidade do modelo.
'''
st.write(f'<div style="text-align: justify"> {md} </div>',unsafe_allow_html=True)


github = f"<a href='https://github.com/MarioLisboaJr'><img src='https://cdn-icons-png.flaticon.com/512/733/733553.png' height='40' width='40'></a>"
linkedin = f"<a href='https://www.linkedin.com/in/mario-lisboa/'><img src='https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg' height='40' width='40'></a>"
portfolio = f"<a href='https://lisboamario.wixsite.com/website'><img src='https://img.icons8.com/clouds/344/external-link.png' height='50' width='50'></a>"

st.markdown(f"<div style='text-align: center'><br><br><br>{github}&ensp;{linkedin}&ensp;{portfolio}<p>{'Mário Lisbôa'}</div>",
            unsafe_allow_html=True)
