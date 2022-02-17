import streamlit as st
import numpy as np
import folium
from folium import IFrame
from streamlit_folium import folium_static
from sys import platform


from catboost import *
import pandas as pd
import geopandas as gpd

#load data
df = gpd.read_file('./sours/selected_df_1.shp', encoding='cp1251')
prepared_data = pd.read_csv('./sours/dw_terr.csv')
model = CatBoostClassifier()
model.load_model('./sours/recsys.cbm')

#виды работ
vid_work = {4:'устройство новых детских площадок, установка игровых комплексов на площадках',
 3:'реконструкция/ремонт детских площадок',
 12:'устройство новых спортивных площадок',
 11:'реконструкция/ремонт спортивных площадок',
 10:'установка/ремонт искусственных неровностей, антипарковочных элементов, ограничителей въезда',
 1:'устройство асфальтных покрытий, дорожной и тропиночной сети',
 15:'размещение/ремонт контейнерных и прочих хозяйственных площадок',
 8:'виды работ, связанные с организацией парковочных пространств',
 2:'виды работ по устройству/ремонту бортового и садового камня',
 0:'виды работ, связанные с организацией площадок для выгула животных',
 13:'виды работ, связанные с организацией уличного освещения',
 5:'прочие виды работ, связанные с установкой/заменой/ремонтом, как архитектурных элементов, так и конструктивных элементов на дворовой территории',
 14:'комплексные работы на дворовой территории',
 7:'виды работ связанные с озеленением территории',
 9:'виды работ, связанные с устройством различных покрытий',
 6:'устройство/ремонт площадок тихого отдыха'}



def show_mapbox_map(lat, lon): #lat, lon
    # Cartodb Positron - базовая карта для приложения
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles='Cartodb Positron', max_bounds=False)
    return m

st.set_page_config(page_title='recomendation_system', initial_sidebar_state="expanded")

st.sidebar.title('Рекомендательная система по выбору вида работ, на дворовых территориях')


st.sidebar.write("[Результат классификации выполненных работ по благоустройству](https://storage.yandexcloud.net/materials/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82_%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D0%B8_%D0%B2%D1%8B%D0%BF%D0%BE%D0%BB%D0%BD%D0%B5%D0%BD%D0%BD%D1%8B%D1%85_%D1%80%D0%B0%D0%B1%D0%BE%D1%82_%D0%BF%D0%BE_%D0%B1%D0%BB%D0%B0%D0%B3%D0%BE%D1%83%D1%81%D1%82%D1%80%D0%BE%D0%B8%CC%86%D1%81%D1%82%D0%B2%D1%83.pdf)")


adress = df['Адресный о'].to_list()
adress.insert(0, '-')
select_adress = st.sidebar.selectbox('Выберите адрес дворовой территории: ', adress)
if select_adress == '-':
    pass
else:
    dwor = df[df['Адресный о'] == select_adress].reset_index()
    user_id = dwor['Номер стро'][0]
    lon = dwor.geometry.centroid.to_list()[0].xy[0][0]
    lat = dwor.geometry.centroid.to_list()[0].xy[1][0]
    st.subheader('Дворовая территория расположенная по адресу: ')
    st.subheader('{}'.format(select_adress))
    # start map
    m = show_mapbox_map(lat, lon)
    dwor_json = dwor.to_crs(epsg='4326').to_json()
    territory = folium.GeoJson(data=dwor_json, name=select_adress)
    territory.add_to(m)
    folium.LayerControl(collapsed=True, show=True).add_to(m)
    folium_static(m, width=850, height=400)
    # stop map
    # start recsys
    predictions = model.predict_proba(prepared_data[prepared_data['user_id'] == user_id])[:, 1]
    preds_df = pd.DataFrame(predictions)
    preds_df = preds_df.sort_values(0, ascending=False)

    #st.subheader(dwor.columns)

    #st.write(dwor['Номер стро'])
    st.write('Ссылка: {}'.format(dwor['Ссылка АИС'][0]))
    st.write('Состав выполненых работ: {}'.format(dwor['Состав раб'][0]))
    st.subheader('Топ рекомендованных видов работ, в порядке убывания: ')
    st.write("(на основе описанной ранее методологии)")
    #st.write(preds_df.reset_index()['index'].head(5).to_list())
    list_pred = preds_df.reset_index()['index'].head(5).to_list()

    if len(list_pred):
        for i, j in enumerate(list_pred):
            st.write(i+1, vid_work[j])
    else:
        st.write("Рекомендации отсутствуют")

    st.stop()
