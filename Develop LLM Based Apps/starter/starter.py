import os
from langchain import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain

inforamtion = """
Riyad Mahrez (en arabe : رياض محرز), né le 21 février 1991 à Sarcelles (France), est un footballeur international algérien. Il évolue au poste d'ailier droit au club saoudien d'Al-Ahli FC depuis 2023.

Formé à l'AAS Sarcelles, il devient professionnel au Quimper Cornouaille FC en 2009, où il n'évolue qu'une saison avant de rejoindre Le Havre. Il y passe trois ans au total, jouant d'abord dans l’équipe réserve puis devenant un habitué de l'équipe première.

Transféré en 2014 à Leicester City, il remporte avec eux la Premier League en 2016. Cette même saison, il devient le premier joueur africain à être désigné joueur de l'année PFA, membre de l'équipe de l'année PFA de Premier League et footballeur algérien de l'année. Il signe ensuite pour Manchester City en 2018, remportant quatre championnats d'Angleterre, une Ligue des champions, deux coupes d'Angleterre, trois Coupes de la Ligue anglaise et un Community Shield.

International algérien à partir de mai 2014, il dispute la Coupe du monde 2014, où la sélection algérienne atteint pour la première fois les huitièmes de finale. En 2019, il gagne la Coupe d'Afrique des nations. 
"""

if __name__ == "__main__":
    summary_prompt = """
        Summarize the following information and provide key facts:

            {inforamtion}
    """
    
    summary_prompt_template = PromptTemplate(input_variables=["information"] , template=summary_prompt)
    
    llm = ChatOpenAI(temperature=0 , model_name="gpt-3.5-turbo")
    
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    
    res = chain.run(inforamtion)
    
    print(res)