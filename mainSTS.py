from altair.vegalite.v4.schema.channels import Key
import streamlit as st
from numpy.lib.function_base import disp
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
#from streamlit_player import st_player
import streamlit.components.v1 as components

header = st.container()
dataset = st.container()
basicstats = st.container()
audiofeatures = st.container()
sentAnalisis = st.container()
Reverse_Search = st.container()
conclusion = st.container()
redesSociales = st.container()



with header:

    header_image = 'https://i.ibb.co/GM1JYrM/header.jpg'
    st.image(header_image, width=800,use_column_width= 'auto')

    st.title('Taylor Swift: dashboard interactivo')
    st.markdown(""" Bienvenidas, bienvenidos, bienvenides al dashboard interactivo de Taylor Swift. 
    Aquí podrás analizar toda la discografía de la güera como nunca antes. 
    Espero te diviertas tanto como yo me divertí armándolo.""")
    
with dataset:
    st.header('Teardrops on my dataframe')
    st.markdown(""" Les platico un poco sobre este **dataframe**""")
    st.markdown("""
    - Contiene toda la discografía de Taylor hasta antes de *Fearless (Taylor's Version)*
    - Los lyrics se obtuvieron desde genius.com
    - Las características de audio de cada canción fueron extraídas desde Spotify 
        - Mood: Danceability, Valence, Energy, Tempo
        - Properties: Loudness, Speechiness
        - Context: Liveness, Acousticness, Genre.""")

    df = pd.read_excel('MasterTaylorSwiftSongs.xlsx')
    df_copy= df.copy(deep=True)
    df.drop(df[df.lyrics.isnull()].index, inplace = True)
    df = df.drop(columns=['sel','main_discography'])

    st.write(df.head(14))

with basicstats:
    st.header('Quick Look')
    st.markdown ('Realizaremos un análisis rápido al álbum y/o canción que elijas')

    sel_col, disp_col = st.columns(2)  

    widget_album = sel_col.selectbox('Elige un álbum',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 0)
    df_album = df[df['album'] == widget_album]
    canciones_album = df_album['title'].unique().tolist()
    canciones_album.append('Todas')
    widget_cancion = disp_col.selectbox('Elige una rola',options=canciones_album,index = (len(canciones_album)-1))

    col1, col2 = st.columns(2)

    with col1:
        st.write('Álbum: ',widget_album)
        st.write('Canciones:', df_album['title'].count())
        st.write('Lanzamiento: ',df_album['year'].max())
        st.write('Bbpm: ',round(df_album['bpm'].mean(),2))
        st.write('Energy: ',round(df_album['nrgy'].mean(),2))
        st.write('Danceability: ',round(df_album['dnce'].mean(),2))
        st.write('dB: ',round(df_album['dB'].mean(),2))
        st.write('Liveness: ',round(df_album['live'].mean(),2))
        st.write('Valence: ',round(df_album['val'].mean(),2))
        st.write('Duration: ',round(df_album['dur'].mean(),2))
        st.write('Acousticness: ',round(df_album['acous'].mean(),2))
        st.write('Speechiness: ',round(df_album['spch'].mean(),2))

    with col2:
           
           
            def set_image(widget_selection):
                if widget_selection == 'Taylor Swift':
                    return 'https://i.ibb.co/7GyHCcF/Taylor-Swift-Albums-Taylor-Swift-Credit-Big-Machine.jpg'
                elif widget_selection == 'Fearless':
                    return 'https://i.ibb.co/0Zs3z4s/Taylor-swift-albums-Taylor-Swift-Fearless-Credit-Big-Machine.jpg'
                elif widget_selection == 'Speak Now':
                    return 'https://i.ibb.co/tbmfnmX/Taylor-Swift-albums-Taylor-Swift-Speak-Now-Credit-Big-Machine.jpg'
                elif widget_selection == 'Red':
                    return 'https://i.ibb.co/ScYfmKw/Taylor-Swift-albums-Taylor-Swift-Red-Credit-Big-Machine.jpg'
                elif widget_selection == '1989':
                    return 'https://i.ibb.co/gzWYdmb/Taylor-Swift-albums-Taylor-Swift-1989-Credit-Big-Machine.jpg'
                elif widget_selection == 'reputation':
                    return 'https://i.ibb.co/KKX3PN8/reputation.jpg'
                elif widget_selection == 'Lover':
                    return 'https://i.ibb.co/Fz8RTTj/Lover.jpg'
                elif widget_selection == 'folklore':
                    return 'https://i.ibb.co/1d23NqV/folklore.jpg'
                elif widget_selection == 'evermore':
                    return 'https://i.ibb.co/FYcV2Cm/evermore.jpg'
                elif widget_selection == 'Fearless (Taylor’s Version)':
                    return 'https://i.ibb.co/vxDsQvm/Fearless.jpg'
                else:
                    return 'https://i.ibb.co/4FtpWM0/Red-TV.jpg'
            
            imagen = set_image(widget_album) 
            st.image(imagen, width=300,use_column_width= 'auto')
            st.markdown("""
             > disclaimer: este botón de rola, aún no sirve para nada, pero próximamente funcionara""")

    with st.expander("Ver explicación:"):
        st.markdown("""
        Todos lo anterior esta expresado en promedios""")
        st.markdown("""
        Entre más grande sea el valor más describe esa característica al álbum """)

with audiofeatures:
    st.header('Deep look')
    st.markdown ("""Realizaremos un análisis más profundo
    sobre las características de audio extraidas de Spotify""")
    
    
    col3, col4 = st.columns(2)

    with col3:
        
        st.write('Álbum: ',widget_album)

        df_albumcor = df_album.drop(columns=['track_n','year'])

        fig, ax = plt.subplots()
        sns.heatmap(df_albumcor .corr(), ax=ax)
        st.write(fig)
    
    with col4:
        st.markdown(""                         "")
        st.markdown("""Las correlaciones nos permiten medir la fuerza y dirección de una relación de dos variables. 
        En el caso de este heatmap, estamos representando las correlaciones entre las características de audio.""")
        st.markdown("""
        Dependiendo el álbum que elijas, podrías encontrar relaciones diferentes.
        Por experiencia las corr más relevantes deberían ser: 
        - acoust vs dnce
        - dnce vs energy""")

    st.subheader('Comparemos las características de 3 canciones: ')
    
    
    col5, col6, col7=st.columns(3)

    with col5:
        widget_album5 = st.selectbox('álbum 1: ',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 0)
        df_album5 = df[df['album'] == widget_album5]
        canciones_album5 = df_album5['title'].unique().tolist()
        canciones_album5.append('Todas')
        widget_cancion5 = st.selectbox('canción 1',options=canciones_album5,index = (len(canciones_album5)-1))
        df_cancion5 = df[df['title'] == widget_cancion5]

        st.write('Bbpm: ',round(df_cancion5['bpm'].mean(),2))
        st.write('Energy: ',round( df_cancion5['nrgy'].mean(),2))
        st.write('Danceability: ',round(df_cancion5['dnce'].mean(),2))
        st.write('dB: ',round(df_cancion5['dB'].mean(),2))
        st.write('Liveness: ',round(df_cancion5['live'].mean(),2))
        st.write('Valence: ',round(df_cancion5['val'].mean(),2))
        st.write('Duration: ',round(df_cancion5['dur'].mean(),2))
        st.write('Acousticness: ',round(df_cancion5['acous'].mean(),2))
        st.write('Speechiness: ',round(df_cancion5['spch'].mean(),2))


    with col6:
        widget_album6 = st.selectbox('álbum 2: ',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 0)
        df_album6 = df[df['album'] == widget_album6]
        canciones_album6 = df_album6['title'].unique().tolist()
        canciones_album6.append('Todas')
        widget_cancion6 = st.selectbox('canción 2',options=canciones_album6,index = (len(canciones_album6)-1))
        df_cancion6 = df[df['title'] == widget_cancion6]

        st.write('Bbpm: ',round(df_cancion6['bpm'].mean(),2))
        st.write('Energy: ',round( df_cancion6['nrgy'].mean(),2))
        st.write('Danceability: ',round(df_cancion6['dnce'].mean(),2))
        st.write('dB: ',round(df_cancion6['dB'].mean(),2))
        st.write('Liveness: ',round(df_cancion6['live'].mean(),2))
        st.write('Valence: ',round(df_cancion6['val'].mean(),2))
        st.write('Duration: ',round(df_cancion6['dur'].mean(),2))
        st.write('Acousticness: ',round(df_cancion6['acous'].mean(),2))
        st.write('Speechiness: ',round(df_cancion6['spch'].mean(),2))
      

    with col7:
        widget_album7 = st.selectbox('álbum 3: ',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 0)
        df_album7 = df[df['album'] == widget_album7]
        canciones_album7 = df_album7['title'].unique().tolist()
        canciones_album7.append('Todas')
        widget_cancion7 = st.selectbox('canción 3',options=canciones_album7,index = (len(canciones_album7)-1))
        df_cancion7 = df[df['title'] == widget_cancion7]

        st.write('Bbpm: ',round(df_cancion7['bpm'].mean(),2))
        st.write('Energy: ',round( df_cancion7['nrgy'].mean(),2))
        st.write('Danceability: ',round(df_cancion7['dnce'].mean(),2))
        st.write('dB: ',round(df_cancion7['dB'].mean(),2))
        st.write('Liveness: ',round(df_cancion7['live'].mean(),2))
        st.write('Valence: ',round(df_cancion7['val'].mean(),2))
        st.write('Duration: ',round(df_cancion7['dur'].mean(),2))
        st.write('Acousticness: ',round(df_cancion7['acous'].mean(),2))
        st.write('Speechiness: ',round(df_cancion7['spch'].mean(),2))

with sentAnalisis:
    st.header('Análisis de la Lyrica')
    st.markdown ("""Como bien lo dijo Taylor en su documental *Miss Americana*, cada artista tiene su nicho propio, 
    y si no fuera por su **story telling** y **lyrics**, ella seria una mas del montón.""")

    col8, col9 =st.columns(2)

    with col8:
    
        diary_image = 'https://i.ibb.co/TBFgsFJ/all-too-well-lyrics.jpg'
        st.image(diary_image, width=200,use_column_width= 'auto')
    
    with col9:
        st.markdown("""No me dejaran mentir, pero muchos nos hicimos fans de la güera por esa habilidad que tiene para describir, lugares, personas y sentimientos universales.""")
        
        st.markdown("""En esta parte intentaremos hacer procesamiento de lenguaje natural a los lyrics de Taylor. Abreviado en ingles como NLP 
        es un campo de las ciencias de la computación, de la inteligencia artificial y de la lingüística que estudia las interacciones entre las computadoras y el lenguaje humano.""")

        st.markdown(""" 
        Disclaimer: Debemos tener en cuenta un chorro de cosas a la hora de hacer análisis, principalmente con:
        - Ironías / Sarcasmos
        - Dobles Sentidos / Metáforas""")   

    st.markdown("""Para armar este análisis tuve que pasar la columna de *lyrics*, por varios procesos de limpieza para que los resultados hicieran sentido.""")    
    st.markdown("""Con ayuda de la librería de nltk, pudimos retirar las **stop words** que son aquellas palabras que no agregan nada al contexto y/o análisis. 
    También quitamos mayúsculas y todos los signos de puntuación.""") 

    import nltk
    nltk.download('punkt')
    from nltk import word_tokenize

    nltk.download('stopwords')
    newStopWords = list(nltk.corpus.stopwords.words('english'))

    import string
    puntuacion = list(string.punctuation)

    pattern = r'''(?x)                  # Flag para iniciar el modo verbose
              (?:[A-Z]\.)+          # Hace match con abreviaciones como U.S.A.
              | \w+(?:-\w+)*        # Hace match con palabras que pueden tener un guión interno
              | \$?\d+(?:\.\d+)?%?  # Hace match con dinero o porcentajes como $15.5 o 100%
              | \.\.\.              # Hace match con puntos suspensivos
              | [][.,;"'?():-_`]    # Hace match con signos de puntuación
    '''

    col10, col11 =st.columns(2)

    with col10:
        widget_album10 = st.selectbox('Escoge un álbum: ',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 0)
        df_album10 = df[df['album'] == widget_album10]

        data_album = df_album10['lyrics'].tolist()

        texto_album = []
        for x in range(0, len(data_album)):
            token_1_album = data_album[x].lower() #convierte a minúsculas
            token_2_album = nltk.regexp_tokenize(token_1_album,pattern) #Quita los patrones definidos arriba y genera tokens
            texto_album.append(token_2_album)

        flatten_album = [ w for l in texto_album for w in l]
        df_2_album = [w for w in flatten_album if w not in newStopWords]
        df_3_album = [w for w in df_2_album if w not in puntuacion]

        vocabulario_album = sorted(set(df_3_album))
        rl_album = len(vocabulario_album)/len(df_3_album)

        from nltk.probability import FreqDist
        fdist_album = FreqDist(df_3_album)

        st.write('Palabras únicas: ',len(vocabulario_album))
        st.write('Complejidad del álbum: ',round(rl_album,3))

        num_palabras_album = st.slider('Cuántas palabras quieres observar?',min_value=5,max_value=20,value=13)

        st.write('las',num_palabras_album,'palabras más repetidas y sus frecuencias')

        fig3, ax3 = plt.subplots()
        fdist_album.most_common(num_palabras_album)
        fw_data_album = pd.DataFrame(fdist_album.items(), columns=['word', 'frequency']).reset_index().sort_values(by='frequency', ascending=False)
        w_plot_album = fw_data_album.head(num_palabras_album)
        ax3.bar(w_plot_album['word'],w_plot_album['frequency'])
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig3)
        
        fig1, ax = plt.subplots()
        wordcloud_album = WordCloud(background_color='white', collocations=False, max_words=30).fit_words(fdist_album)
        plt.imshow(wordcloud_album, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot(fig1)
        
    with col11:
        canciones_album10 = df_album10['title'].unique().tolist()
        widget_cancion10 = st.selectbox('Escoge una canción:',options=canciones_album10,index = 0)
        df_cancion10 = df[df['title'] == widget_cancion10]

        data_cancion = df_cancion10['lyrics'].tolist()

        texto_cancion = []
        for x in range(0, len(data_cancion)):
            token_1_cancion = data_cancion[x].lower() #convierte a minúsculas
            token_2_cancion = nltk.regexp_tokenize(token_1_cancion,pattern) #Quita los patrones definidos arriba y genera tokens
            texto_cancion.append(token_2_cancion)

        flatten_cancion = [ w for l in texto_cancion for w in l]
        df_2_cancion = [w for w in flatten_cancion if w not in newStopWords]
        df_3_cancion = [w for w in df_2_cancion if w not in puntuacion]

        vocabulario_cancion = sorted(set(df_3_cancion))
        rl_cancion = len(vocabulario_cancion)/len(df_3_cancion)

        from nltk.probability import FreqDist
        fdist_cancion = FreqDist(df_3_cancion)

        st.write('Palabras únicas: ',len(vocabulario_cancion))
        st.write('Complejidad de la canción: ',round(rl_cancion,3))

        num_palabras_cancion = st.slider('Cuántas palabras quieres ver?',min_value=5,max_value=20,value=13)

        st.write('las',num_palabras_cancion,'palabras más repetidas y sus frecuencias')

        fig4, ax4 = plt.subplots()
        fdist_cancion.most_common(num_palabras_cancion)
        fw_data_cancion = pd.DataFrame(fdist_cancion.items(), columns=['word', 'frequency']).reset_index().sort_values(by='frequency', ascending=False)
        w_plot_cancion = fw_data_cancion.head(num_palabras_cancion)
        ax4.bar(w_plot_cancion['word'],w_plot_cancion['frequency'])
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig4)

        fig2, ax = plt.subplots()
        wordcloud_cancion = WordCloud(background_color='white', collocations=False, max_words=30).fit_words(fdist_cancion)
        plt.imshow(wordcloud_cancion, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        st.pyplot(fig2)
        
    with st.expander("Cómo interpretar esto?:"):
            st.markdown("""
            - Palabras únicas: es el total de palabras que no se repiten""")
            st.markdown("""
            - Complejidad: se refiere al numero de palabras únicas entre el total de palabras """)
            st.markdown("""
            Siempre es muy interesante ver que palabras son las que más se repiten y cómo influyen en el sentimiento.""")

    st.subheader('Análisis de sentimiento')
    st.markdown("""Para este análisis utilizamos la variable ***compound*** que es un promedio ponderado entre las calificaciones otorgadas a ***positivo,neutro,negativo*** , 
    para así encontrar el sentimiento general de las canciones""")
        
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyser_album = SentimentIntensityAnalyzer()

    def sentiment_analyzer_scores_album(lyric):
            score_album = analyser_album.polarity_scores(lyric)
            comp_album = score_album['compound']
            return comp_album

    df_album10['sentiment'] = df_album10['lyrics'].apply(lambda x: sentiment_analyzer_scores_album(x))

    fig7, ax7 = plt.subplots()
    plt.scatter(df_album10['sentiment'], df_album10['title'])
    ax7 = plt.gca()
    ax7.set_ylim(ax7.get_ylim()[::-1])
    st.pyplot(fig7)
    
with Reverse_Search:
    st.header('Indagemos un poco más...')
    st.markdown(""" ¿Alguna vez te has preguntado en que canciones Taylor usa una misma palabra? y ¿cómo están relacionadas? """)
    df_busqueda_0 = df_copy[['lyrics','title','main_discography']]
    df_busqueda= df_busqueda_0[df_busqueda_0['main_discography'] == 1]
    busqueda_inv = st.text_input("Introduce una palabra a buscar en las canciones: ", value="blue")
    df_busqueda_2 = df_busqueda[df_busqueda.apply(lambda row: row.astype(str).str.contains(busqueda_inv).any(), axis=1)]['title'].to_list()
    if len(df_busqueda_2) == 0:
        st.markdown('No se encontraron resultados')
    else:
        st.markdown('Resultados de la búsqueda:') #ESCRIBIR EXPLICACION RESULTADOS
        st.write(df_busqueda_2)


with conclusion:
    st.header('¿Qué aprendimos?')
    st.markdown ("""Espero que después de jugar con este dashboard hayas podido encontrar revelaciones interesantes sobre la música de Taylor,
    también se, que como los grandes swifties que somos, muchas otras cosas no nos hicieron sentido... i mean como es que ***All too well*** y ***...Redy for it?***
    tienen casi la misma bailabilidad???""")
    st.markdown(""" Este tipo de modelos son muy nuevos, sin embargo, los utilizamos en nuestras vidas diarias TODO el tiempo, algunos ejemplos son:""")
    st.markdown (""" 
            - Asistentes virtuales 
            - Predictores de texto
            - Traducciones
            - Filtros de contenido
            - Análisis de sentimientos
            - o en mi caso analizar las canciones de mi artista favorita...""")

    st.markdown("""Aunque amo ver las canciones de Taylor traducidas a datos y números, nunca lograran describir lo que realmente te hacen sentir las canciones, los suspiros,
    las entonaciones, las constantes metáforas y relaciones a ciertas horas del día o colores específicos, a las personas que te recuerda, en fin, podría pasar hoooooras hablando sobre cada una de estas canciones""")

    
    st.markdown("""Lo bueno es que existe una solución para estos problemas, *swifting podcast* """)
    st.markdown("""un podcast donde analizamos canción por canción, verso por verso, palabra por palabra de la música de Taylor Swift (casi casi como máquinas de AI)""")

   
    st.subheader('Revisa si tu canción fav ya fue analizada por swifting podcast.')
    
    widget_podcast = st.selectbox('álbum fav: ',options=['Taylor Swift','Fearless','Speak Now','Red','1989','reputation','Lover','folklore','evermore','Fearless (Taylor´s Version)','Red (Taylor´s Version)'],index = 8)
    df_podcast = df[df['album'] == widget_podcast]
    canciones_podcast= df_podcast['title'].unique().tolist()
    widget_cancion_podcast = st.selectbox('canción fav: ',options=canciones_podcast)
    
    episodio_podcast = df_podcast[df_podcast['title'] == widget_cancion_podcast].reset_index().loc[0, 'URL']

     
    components.html(episodio_podcast)
    st.markdown("""Si tu canción favorita no ha sido analizada, no te preocupes puedes empezar por las primeras temporadas o escuchando los episodios bonus.""")
    st.write("[Escucha en cualquier plataforma](https://swifting.captivate.fm/listen)")

with redesSociales:
    st.markdown(""" Tienes dudas, comentarios o buscas más contenido?? Contáctame [@nataliavazquez](https://linktr.ee/nataliavazquez)""" )

