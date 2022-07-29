import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import owlready2

def printProject(projects):

    for p in projects:

        if onto.Project in p.is_a:
            st.write(p.name)
            for rel in onto['PR.hasDescription'].get_relations():
                if rel[0] == p:
                    st.write("Description: %s" % rel[1])
            break


def lookForProject(specie, phenotype, tissue):
        projectsPh = []
        for rel in onto['SPR.hasDisease'].get_relations():
            if rel[1].name == qPh:
                projectsPh.append(rel[0])


        projectsSp = []
        for rel in onto['SPR.hasSpecie'].get_relations():
            if rel[1].name == qSp:
                projectsSp.append(rel[0])

        projectsTs = []
        for rel in onto['SPR.hasOrganismPart'].get_relations():
            if rel[1].name == qTs:
                projectsTs.append(rel[0])

        projects = list(set(projectsPh) & set(projectsSp) & set(projectsTs))
        if projects:

            st.write('Result:')
            printProject(projects)
        else:
            projectsControl = []
            for rel in onto['SPR.hasDisease'].get_relations():
                if rel[1].name == 'Control':
                    projectsControl.append(rel[0])

            projects = list(set(projectsControl) & set(projectsSp) & set(projectsTs))
            if projects:
                st.write('Result:')
                printProject(projects)
            else:
                projects = list(set(projectsPh) & set(projectsSp))
                if projects:
                    st.write('Result:')
                    printProject(projects)
                else:
                    st.write("No results found.")

st.title("Playing Room for Models")

qSpText = ""
qPhText = ""
qTsText = ""

m = st.selectbox(
     'Select the model to use (load could take a while):',
     ('BioBert-FineTune-Base-multitask', 'BioBert-FineTune-Base-multitask-suffle'))

model = SentenceTransformer(m)

onto = owlready2.get_ontology('../out_repositoriev6.owl').load()

query = st.text_input('Write the question here:', 'What is the most common disease in Zebra Fish?')

species = ['Anopheles Gambiae',
 'Arabidopsis Thaliana',
 'Caenorhabditis Elegans',
 'Callithrix Jacchus',
 'Danio Rerio',
 'Drosophila Melanogaster',
 'Gallus Gallus',
 'Homo Sapiens',
 'Mus Musculus',
 'Plasmodium Berghei',
 'Plasmodium Falciparum',
 'Rattus Norvegicus',
 'Saccharomyces Cerevisiae',
 'Schistosoma Mansoni']

phenotype = ['Acoustic Neuroma',
 'Arthritis',
 'Breast Cancer',
 'COVID-19',
 'Cataract',
 'Fibrosis',
 'HIV',
 'Hepatitis C',
 'Kidney Cancer',
 'Lyme Disease',
 'Neoplasm',
 'Pancreatic Cancer',
 'Parkinsons Disease',
 'Prostate Cancer',
 'Stroke Disorder']

tissue = ['Aorta',
 'Blood',
 'Brain',
 'Colon',
 'Dermis',
 'Endoderm',
 'Eye',
 'Gastrointestinal System',
 'Glial Cells',
 'Hematopoietic System',
 'Hippocampus',
 'Induced Pluripotent',
 'Larynx',
 'Leukocyte',
 'Liver']


embeddingsSp = model.encode(species, convert_to_tensor=True)
embeddingsPh = model.encode(phenotype, convert_to_tensor=True)
embeddingsTs = model.encode(tissue, convert_to_tensor=True)

if st.button('Send'):

    col1, col2, col3 = st.columns(3)

    embeddings1 = model.encode(query, convert_to_tensor=True)

    with col1:
        st.header("Specie")
        #Compute cosine-similarits
        cosine_scores = util.cos_sim(embeddings1, embeddingsSp)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(0, len(species)):
            pairs.append({'index': i, 'score': cosine_scores[0][i]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top1 = pairs[0]
        top3 = pairs[0:3]
        top5 = pairs[0:5]

        predictedSp = species[top1['index']]

        st.write("Predicted: %s" % predictedSp)

        ontoInstanceSp = onto[predictedSp.replace(' ','')]
        ontoClassSp = ontoInstanceSp.is_a

        for prop in ontoInstanceSp.get_properties():
            pDf = pd.DataFrame()
            related = []
            first = True
            for rel in prop.get_relations():

                if rel[1] == prop[ontoInstanceSp][0]:

                    if first:
                        st.write("Ohter Species regarding %s" % rel[1].name)
                    first = False

                    #st.write("\t- %s" % rel[0].name)
                    related.append(rel[0].name)
            if related:
                pDf[prop[ontoInstanceSp][0].name] = related
                st.dataframe(pDf)
                

    with col2:
        st.header("Phenotype")

        #Compute cosine-similarits
        cosine_scores = util.cos_sim(embeddings1, embeddingsPh)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(0, len(phenotype)):
            pairs.append({'index': i, 'score': cosine_scores[0][i]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top1 = pairs[0]
        top3 = pairs[0:3]
        top5 = pairs[0:5]

        predictedPh = phenotype[top1['index']]

        st.write("Predicted: %s" % predictedPh)

        ontoInstancePh = onto[predictedPh.replace(' ','')]
        ontoClassPh = ontoInstancePh.is_a

        for c in ontoClassPh:

            pDf = pd.DataFrame()
            related = []
            first = True
            for i in c.instances():
                if i != ontoInstancePh:
                    if first:
                        st.write("Other types of %s" % c.name)
                        first = False
                    #st.write("\t- %s" % i.name)
                    related.append(i.name)
            if related:
                pDf[c.name] = related
                st.dataframe(pDf)
        
        for prop in ontoInstancePh.get_properties():
            pDf = pd.DataFrame()
            related = []
            first = True
            for rel in prop.get_relations():

                if rel[1] == prop[ontoInstancePh][0]:

                    if first:
                        st.write("Ohter Phenotypes regarding %s" % rel[1].name)
                        first = False

                    if rel[0] != ontoInstancePh:
                        st.write("\t- %s" % rel[0].name)
                        related.append(rel[0].name)
                        
            if related:
                pDf[prop[ontoInstancePh][0].name] = related
                st.dataframe(pDf)

    with col3:
        st.header("Tissue")

        #Compute cosine-similarits
        cosine_scores = util.cos_sim(embeddings1, embeddingsTs)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(0, len(tissue)):
            pairs.append({'index': i, 'score': cosine_scores[0][i]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top1 = pairs[0]
        top3 = pairs[0:3]
        top5 = pairs[0:5]

        predictedTs = tissue[top1['index']]

        st.write("Predicted: %s" % predictedTs)

        ontoInstanceTs = onto[predictedTs.replace(' ','')]
        ontoClassTs = ontoInstanceTs.is_a

        
        for c in ontoClassTs:
            pDf = pd.DataFrame()
            related = []
            first = True
            for i in c.instances(): 
                if i != ontoInstanceTs:
                    if first:
                        st.write("Other types of %s" % c.name)
                        first = False
                    #st.write("\t- %s" % i.name)
                    related.append(i.name)
            if related:
                pDf[c.name] = related
                st.dataframe(pDf)

        for prop in ontoInstanceTs.get_properties():
            pDf = pd.DataFrame()
            related = []

            first = True
            for rel in prop.get_relations():

                if rel[1] == prop[ontoInstanceTs][0]:

                    if first:
                        st.write("Ohter Tissue regarding %s" % rel[1].name)
                        first = False
                    if rel[0] != ontoInstanceTs:
                        #st.write("\t- %s" % rel[0].name)
                        related.append(rel[0].name)
            if related:
                pDf[prop[ontoInstanceTs][0].name] = related
                st.dataframe(pDf)
    
    qSpText = ontoInstanceSp.name
    qPhText = ontoInstancePh.name
    qTsText = ontoInstanceTs.name


col1, col2, col3 = st.columns(3)


with col1:
        qSp = st.text_area('Specie Search:', qSpText)

with col2:
        qPh = st.text_area('Phenotype Search:', qPhText)

with col3:
        qTs = st.text_area('Tissue Search:', qTsText)

if st.button('Look for Projects:'):
    print(qSp)
    print(qPh)
    print(qTs)
    lookForProject(qSp, qPh, qTs)