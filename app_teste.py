import streamlit as st
import pandas as pd
import networkx as nx
from community import community_louvain
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from pyvis.network import Network
import streamlit.components.v1 as components
import itertools
import pdfplumber

# ==================== CONFIGURAÇÕES INICIAIS ====================
st.set_page_config(page_title="Dashboard Auxílio Estudantil", layout="wide")

PESOS_SIMILARIDADE = [0.20, 0.15, 0.12, 0.12, 0.10, 0.10, 0.09, 0.09, 0.03]
PESOS_NECESSIDADE = {
    'moradia_aluno': 20, 'moradia_familia': 15, 'renda_classe': 12, 
    'despesa_classe': 10, 'bens_familiares_classe': 9, 'transporte': 3
}

# ==================== FUNÇÕES DE DADOS E MATEMÁTICA ====================
@st.cache_data
def carregar_dados(arquivo_upload):
    nome_arquivo = arquivo_upload.name.lower()
    
    # Leitura de múltiplos formatos
    if nome_arquivo.endswith('.csv'):
        df = pd.read_csv(arquivo_upload, sep=None, engine='python')
    elif nome_arquivo.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(arquivo_upload)
    elif nome_arquivo.endswith('.pdf'):
        linhas_tabela = []
        with pdfplumber.open(arquivo_upload) as pdf:
            for pagina in pdf.pages:
                tabela = pagina.extract_table()
                if tabela:
                    # Limpa quebras de linha e células nulas que o PDF pode gerar
                    tabela_limpa = [[str(celula).replace('\n', ' ').strip() if celula else "" for celula in linha] for linha in tabela]
                    linhas_tabela.extend(tabela_limpa)
        
        if not linhas_tabela:
            st.error("Nenhuma tabela foi encontrada na leitura do PDF.")
            return pd.DataFrame(), {}
            
        df = pd.DataFrame(linhas_tabela[1:], columns=linhas_tabela[0])
    else:
        return pd.DataFrame(), {}
        
    df = df.fillna(0)
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    
    # Processamento e mapeamento das colunas
    if 'id_discente' not in df.columns and 'ID' in df.columns:
        linhas_processadas = []
        for _, row in df.iterrows():
            id_aluno = str(row.get('ID', '')).strip()
            if not id_aluno or id_aluno == '0' or id_aluno == '0.0' or id_aluno.upper() == 'ID': 
                continue # Pula cabeçalhos repetidos no PDF ou linhas vazias
            
            moradia_texto = str(row.get('Situação da moradia do aluno', '')).upper()
            if 'ALUGADO' in moradia_texto and 'SOZINHO' in moradia_texto: moradia_aluno = 1
            elif 'DIVIDE' in moradia_texto: moradia_aluno = 2
            elif 'FAMÍLIA' in moradia_texto or 'PARENTES' in moradia_texto: moradia_aluno = 3
            elif 'PRÓPRIO' in moradia_texto or 'CEDIDO' in moradia_texto: moradia_aluno = 4
            else: moradia_aluno = 2 
                
            moradia_fam_texto = str(row.get('Situação da moradia do grupo familiar', '')).upper()
            if 'ALUGADA' in moradia_fam_texto: moradia_familia = 1
            elif 'FINANCIADA' in moradia_fam_texto or 'PAGAMENTO' in moradia_fam_texto: moradia_familia = 2
            elif 'QUITADA' in moradia_fam_texto: moradia_familia = 3
            elif 'CEDIDA' in moradia_fam_texto or 'HERANÇA' in moradia_fam_texto: moradia_familia = 4
            else: moradia_familia = 3 
                
            renda_texto = str(row.get('Renda familiar por pessoa', ''))
            if 'entre R$0,00 e R$706,00' in renda_texto:
                renda_valor = 350; renda_classe = 1
            elif 'entre R$706,01 e R$1.412,00' in renda_texto:
                renda_valor = 1050; renda_classe = 2
            elif 'entre R$1.412,01 e R$2.118,00' in renda_texto:
                renda_valor = 1750; renda_classe = 3
            elif 'Acima' in renda_texto:
                renda_valor = 2500; renda_classe = 4
            else:
                try:
                    # Tenta converter caso o PDF traga apenas o número
                    renda_valor = float(str(renda_texto).replace('R$', '').replace('.', '').replace(',', '.').strip())
                    renda_classe = 1 if renda_valor <= 706 else 2 if renda_valor <= 1412 else 3 if renda_valor <= 2118 else 4
                except ValueError:
                    renda_valor = 1000; renda_classe = 2

            despesa_valor = int(renda_valor * 0.8)
            despesa_classe = renda_classe
            bens_valor = 50000
            bens_classe = 2
            transporte = 3
            
            linhas_processadas.append({
                'id_discente': id_aluno,
                'Qual a situação da MORADIA DO ALUNO?': moradia_aluno,
                'Qual a situação da MORADIA DO GRUPO FAMILIAR?': moradia_familia,
                'Renda per capita': renda_valor,
                'classes (Renda per capita)': renda_classe,
                'Despesas per capita': despesa_valor,
                'classes (Despesas per capita)': despesa_classe,
                'Valor Total dos bens familiares': bens_valor,
                'classes (Valor Total dos bens familiares)': bens_classe,
                'Qual o principal meio de transporte que você utiliza para vir até a Universidade?': transporte
            })
        df = pd.DataFrame(linhas_processadas)
        
    alunos = df.to_dict('records')
    dic_alunos = {str(a['id_discente']): a for a in alunos}
    return df, dic_alunos

@st.cache_resource
def calcular_todas_similaridades(dic_alunos):
    todas_rendas = [a['Renda per capita'] for a in dic_alunos.values()]
    todos_bens = [a['Valor Total dos bens familiares'] for a in dic_alunos.values()]
    todas_despesas = [a['Despesas per capita'] for a in dic_alunos.values()]
    
    dif_max_renda = max(todas_rendas) - min(todas_rendas) or 1
    dif_max_bens = max(todos_bens) - min(todos_bens) or 1
    dif_max_despesa = max(todas_despesas) - min(todas_despesas) or 1
    
    cache_sim = []
    ids = list(dic_alunos.keys())
    
    for id1, id2 in itertools.combinations(ids, 2):
        a1, a2 = dic_alunos[id1], dic_alunos[id2]
        
        sim_ma = 1 if a1['Qual a situação da MORADIA DO ALUNO?'] == a2['Qual a situação da MORADIA DO ALUNO?'] else 0
        sim_mf = 1 if a1['Qual a situação da MORADIA DO GRUPO FAMILIAR?'] == a2['Qual a situação da MORADIA DO GRUPO FAMILIAR?'] else 0
        sim_rc = 1 if a1['classes (Renda per capita)'] == a2['classes (Renda per capita)'] else 0
        sim_dc = 1 if a1['classes (Despesas per capita)'] == a2['classes (Despesas per capita)'] else 0
        sim_bc = 1 if a1['classes (Valor Total dos bens familiares)'] == a2['classes (Valor Total dos bens familiares)'] else 0
        sim_tr = 1 if a1['Qual o principal meio de transporte que você utiliza para vir até a Universidade?'] == a2['Qual o principal meio de transporte que você utiliza para vir até a Universidade?'] else 0
        
        sim_renda = 1 - (abs(a1['Renda per capita'] - a2['Renda per capita']) / dif_max_renda)
        sim_despesa = 1 - (abs(a1['Despesas per capita'] - a2['Despesas per capita']) / dif_max_despesa)
        sim_bens = 1 - (abs(a1['Valor Total dos bens familiares'] - a2['Valor Total dos bens familiares']) / dif_max_bens)
        
        similaridades = [sim_ma, sim_mf, sim_renda, sim_rc, sim_despesa, sim_dc, sim_bens, sim_bc, sim_tr]
        sim_total = sum(p * s for p, s in zip(PESOS_SIMILARIDADE, similaridades))
        
        cache_sim.append((id1, id2, sim_total))
        
    return cache_sim

def pontuacao_necessidade(aluno):
    ma = aluno['Qual a situação da MORADIA DO ALUNO?']
    pts_ma = 6 if ma == 1 else 4 if ma == 2 else 2 if ma == 3 else 0
    
    mf = aluno['Qual a situação da MORADIA DO GRUPO FAMILIAR?']
    pts_mf = 6 if mf == 1 else 4.5 if mf == 2 else 3 if mf == 3 else 1.5 if mf == 4 else 0
    
    calc_c = lambda v: 6 if v == 1 else 5.4 if v == 2 else 4.8 if v == 3 else 4.2 if v == 4 else 0
    pts_rc = calc_c(aluno['classes (Renda per capita)'])
    pts_dc = calc_c(aluno['classes (Despesas per capita)'])
    pts_bc = calc_c(aluno['classes (Valor Total dos bens familiares)'])
    
    tr = aluno['Qual o principal meio de transporte que você utiliza para vir até a Universidade?']
    pts_tr = 6 if tr == 1 else 5 if tr == 2 else 4 if tr == 3 else 3 if tr == 4 else 2 if tr == 5 else 1 if tr == 6 else 0
    
    return (pts_ma * PESOS_NECESSIDADE['moradia_aluno']) + \
           (pts_mf * PESOS_NECESSIDADE['moradia_familia']) + \
           (pts_rc * PESOS_NECESSIDADE['renda_classe']) + \
           (pts_dc * PESOS_NECESSIDADE['despesa_classe']) + \
           (pts_bc * PESOS_NECESSIDADE['bens_familiares_classe']) + \
           (pts_tr * PESOS_NECESSIDADE['transporte'])

# ==================== INÍCIO DA INTERFACE ====================
st.title("🎓 Sistema Inteligente de Auxílio Estudantil")
st.markdown("Plataforma de clusterização de alunos baseada em grafos de similaridade socioeconômica.")

# Menu Lateral
with st.sidebar:
    st.header("⚙️ Parâmetros do Modelo")
    limiar = st.slider("Limiar de Similaridade", min_value=0.0, max_value=1.0, value=0.89, step=0.01)
    
    # É EXATAMENTE AQUI QUE VOCÊ VAI ALTERAR:
    resolucao = st.number_input(
        "Resolução do Louvain", 
        value=1.38, 
        step=0.1,
        help="A 'Resolução' ajusta o tamanho dos grupos. Valores altos fragmentam os alunos em dezenas de grupos pequenos. Se a análise estiver muito dividida, reduza para 1.0 ou 0.8 para forçar a criação de blocos mais amplos."
    )
    
    st.info("💡 **Dica:** Se o grafo formar dezenas de grupos pequenos (G-40, G-50...), diminua a 'Resolução do Louvain' para 1.0 ou 0.8 para forçar o algoritmo a formar grupos maiores.")

# Declaração das Abas
aba_intro, aba_upload, aba1, aba2, aba3 = st.tabs([
    "📖 Sobre o Sistema",
    "📂 Inserção de Dados", 
    "📊 Visão Geral (Grafos)", 
    "🧮 Auditoria Matemática", 
    "📋 Tabela de Prioridades"
])

# --- ABA 0: SOBRE O SISTEMA ---
with aba_intro:
    st.subheader("Bem-vindo ao Sistema Inteligente de Auxílio Estudantil")
    st.markdown("""
    Este dashboard foi desenvolvido para otimizar a análise de vulnerabilidade socioeconômica, auxiliando a tomada de decisão na distribuição de recursos.
    * **📂 Inserção de Dados:** Faça o upload da planilha da secretaria (Excel, CSV ou PDF tabular).
    * **📊 Visão Geral (Grafos):** Exibe um mapa visual das conexões entre os estudantes. Alunos semelhantes são agrupados nas mesmas "Comunidades" (cores) através do algoritmo de *Louvain*.
    * **🧮 Auditoria Matemática:** Comprova a eficácia dos grupos formados comparando com *K-Means*.
    * **📋 Tabela de Prioridades:** Ranking final listando os alunos do maior para o menor grau de vulnerabilidade.
    """)

# --- ABA DE UPLOAD ---
with aba_upload:
    st.subheader("Passo 1: Alimentar o Sistema")
    arquivo_upload = st.file_uploader("Faça upload do arquivo (CSV, XLSX, XLS ou PDF)", type=["csv", "xlsx", "xls", "pdf"])

if arquivo_upload is not None:
    df, dic_alunos = carregar_dados(arquivo_upload)
    
    if df.empty or not dic_alunos:
        st.error("Erro ao processar o arquivo. Verifique se o documento contém os cabeçalhos esperados e uma tabela bem formatada.")
    else:
        cache_sim = calcular_todas_similaridades(dic_alunos)

        G = nx.Graph()
        G.add_nodes_from(dic_alunos.keys())
        for id1, id2, sim in cache_sim:
            if sim >= limiar:
                G.add_edge(id1, id2, weight=sim)

        qtd_conexoes = G.number_of_edges()
        grafo_seguro = 0 < qtd_conexoes <= 50000 

        if grafo_seguro:
            particao = community_louvain.best_partition(G, resolution=resolucao)
        else:
            particao = {}

        resultados = []
        for id_aluno, dados in dic_alunos.items():
            pts = pontuacao_necessidade(dados)
            if grafo_seguro and id_aluno in particao:
                grupo = particao[id_aluno]
            elif not grafo_seguro:
                grupo = "Pendente (Limiar Inseguro)"
            else:
                grupo = "Isolado"
                
            resultados.append({
                "ID Aluno": id_aluno,
                "Pontuação de Necessidade": round(pts, 2),
                "Renda Per Capita": f"R$ {dados['Renda per capita']:.2f}",
                "Comunidade (Grupo)": grupo
            })
            
        df_resultados = pd.DataFrame(resultados)
        df_resultados = df_resultados.sort_values(by="Pontuação de Necessidade", ascending=False).reset_index(drop=True)

        with aba_upload:
            if not df_resultados.empty:
                st.divider()
                aluno_top1 = df_resultados.iloc[0]
                st.success(f"🏆 **ALUNO COM MAIOR PRIORIDADE DE AUXÍLIO:** ID **{aluno_top1['ID Aluno']}** — Pontuação de Vulnerabilidade: **{aluno_top1['Pontuação de Necessidade']}** | {aluno_top1['Renda Per Capita']}")

        # --- ABA 1: GRAFOS INTERATIVOS ---
        with aba1:
            st.subheader(f"Grafo de Similaridade (Limiar: {limiar})")
            
            if qtd_conexoes == 0:
                st.warning("Nenhuma conexão encontrada para este limiar. Tente diminuir o valor na barra lateral.")
            elif not grafo_seguro:
                st.error(f"⚠️ Alerta de Sobrecarga extrema: O grafo gerou **{qtd_conexoes} conexões**.")
            else:
                st.success(f"Desenhando {qtd_conexoes} conexões com layout calculado no Python...")
                
                cores = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
                         "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", 
                         "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", 
                         "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080"]
                
                comunidades_unicas = sorted(list(set(particao.values())))
                
                legenda_html = "<div style='max-height: 120px; overflow-y: auto; padding: 12px; border: 1px solid rgba(128, 128, 128, 0.3); border-radius: 8px; margin-bottom: 15px;'>"
                legenda_html += "<span style='font-weight: bold; font-size: 14px; margin-bottom: 10px; display: block;'>🎨 Legenda de Grupos Encontrados:</span>"
                legenda_html += "<div style='display: flex; flex-wrap: wrap; gap: 8px;'>"
                
                for c in comunidades_unicas:
                    cor = cores[c % len(cores)]
                    legenda_html += f"<div style='display: flex; align-items: center; background-color: rgba(128, 128, 128, 0.1); padding: 4px 12px; border-radius: 15px; font-size: 13px; border: 1px solid rgba(128, 128, 128, 0.3);'><div style='width: 12px; height: 12px; background-color: {cor}; border-radius: 50%; margin-right: 8px;'></div>Grupo {c}</div>"
                
                legenda_html += "</div></div>"
                st.markdown(legenda_html, unsafe_allow_html=True)

                posicoes = nx.spring_layout(G, k=0.5, seed=42)
                net = Network(height="650px", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources='remote')
                
                for no in G.nodes():
                    if G.degree(no) > 0: 
                        comunidade = particao.get(no, 0)
                        aluno_dados = dic_alunos[no]
                        info = f"ID: {no}\nRenda: R$ {aluno_dados['Renda per capita']:.2f}\nComunidade: {comunidade}"
                        cor = cores[comunidade % len(cores)]
                        x, y = posicoes[no]
                        net.add_node(no, label=str(no), title=info, color=cor, size=20, x=float(x*1500), y=float(y*1500))
                        
                for id1, id2, dados in G.edges(data=True):
                    net.add_edge(id1, id2, value=dados['weight'], title=f"Similaridade: {dados['weight']:.2f}", color="rgba(150, 150, 150, 0.1)")
                    
                net.toggle_physics(False)
                net.save_graph("grafo.html")
                
                with open("grafo.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=670)

        # --- ABA 2: AUDITORIA MATEMÁTICA ---
        with aba2:
            st.subheader("Comparação de Algoritmos de Clusterização")
            
            if qtd_conexoes == 0:
                st.warning("Gere um grafo com conexões para visualizar as métricas matemáticas.")
            elif not grafo_seguro:
                st.warning("O número de conexões é muito alto. Aumente o limiar para realizar a auditoria.")
            else:
                col1, col2 = st.columns(2)
                modularity = community_louvain.modularity(particao, G)
                qtd_comunidades = len(set(particao.values()))
                
                with col1:
                    st.info("### Algoritmo: Louvain (Grafos)")
                    st.metric("Comunidades Formadas", qtd_comunidades)
                    st.metric("Índice de Modularidade", f"{modularity:.4f}")
                    
                features = df[['Qual a situação da MORADIA DO ALUNO?', 
                               'Qual a situação da MORADIA DO GRUPO FAMILIAR?', 
                               'Renda per capita', 'Despesas per capita', 
                               'Valor Total dos bens familiares', 
                               'Qual o principal meio de transporte que você utiliza para vir até a Universidade?']].copy()
                
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                if qtd_comunidades > 1:
                    kmeans = KMeans(n_clusters=qtd_comunidades, random_state=42, n_init='auto')
                    kmeans_labels = kmeans.fit_predict(features_scaled)
                    
                    try:
                        sil_score_kmeans = silhouette_score(features_scaled, kmeans_labels)
                    except ValueError:
                        sil_score_kmeans = 0.0
                    
                    louvain_labels_lista = [particao.get(str(id_a), 0) for id_a in df['id_discente']]
                    try:
                        sil_score_louvain = silhouette_score(features_scaled, louvain_labels_lista)
                    except ValueError:
                        sil_score_louvain = 0.0
                    
                    with col2:
                        st.success("### Algoritmo: K-Means (ML)")
                        st.metric("Clusters Formados", qtd_comunidades)
                        st.metric("Coeficiente de Silhueta (K-Means)", f"{sil_score_kmeans:.4f}")
                        
                    st.divider()
                    st.markdown(f"**Conclusão da Análise:** O Coeficiente de Silhueta do modelo em grafos (Louvain) foi de **{sil_score_louvain:.4f}**.")
                else:
                    st.info("Apenas 1 comunidade foi identificada. São necessários pelo menos 2 grupos distintos para calcular as métricas de Silhueta.")

        # --- ABA 3: TABELA DE PRIORIDADES ---
        with aba3:
            st.subheader("Ranking de Prioridade para Auxílio")
            st.dataframe(df_resultados, use_container_width=True, height=500)

else:
    with aba1:
        st.info("👈 Aguardando o upload do arquivo na aba 'Inserção de Dados'.")
    with aba2:
        st.info("👈 Aguardando o upload do arquivo na aba 'Inserção de Dados'.")
    with aba3:
        st.info("👈 Aguardando o upload do arquivo na aba 'Inserção de Dados'.")