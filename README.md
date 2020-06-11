# Projeto-AceleraDev-Data-Science

O objetivo deste projeto é constuir um sistema para fazer a recomendação de possíveis clientes ao usuário com base em sua lista atual de clientes.

## Estrutura do projeto

O projeto é dividido em duas principais partes, o pré-processamento, que é independente ao usuário, e o sistema de recomendação em si.

![Fluxograma do projeto](https://user-images.githubusercontent.com/46359888/84395773-84f0a000-abd4-11ea-96ab-ae17571e129d.png)



### Pré-processamento

No notebook "Analise dos dados.ipynb" é realizado o primeiro tratamento dos dados, nele os dados contidos na pasta "Raw_data" foram selecionadas e tratadas as variáveis a serem utilizadas. Em "generate_dataset.py" os dados previamente tratados são convertidos para arquivos hdf5 e separados para o treinamento do autoencoder.

O script "main autoencoder.py" realiza o treino dos autoencoders, estes são redes neurais que utilizam dropout como forma de fazer uma aproximação bayesiana. Estes autoencoders tem dois objetivos, além de depois serem usados para fazer a redução de dimensionalidade eles devem fazer a regressão dos dados faltantes em sua saída, para isso foi adicionada uma variável booleana para cada variável que indica quando a regressão é necessária e para o treino das redes dados faltantes foram simulados. 

O autoencoder que possuí o menor erro e maior compressão é selecionado manualmente para fazer a codificação em "encode.py", neste script todos os dados do mercado são processados pela primeira parte do autoencoder e são salvos na pasta "Data" com o código do autoencoder utilizado.

### Sistema de recomendação ao usuário

Com base no portifólio do usuário o notebook "Main.ipynb" utiliza o script "Deep_one_Class.py" para realizar o treino de uma rede neural, que também usa dropout como forma de fazer uma aproximação bayesiana, para fazer a recomendação. A rede é 
