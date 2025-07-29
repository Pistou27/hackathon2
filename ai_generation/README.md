Pipeline de Génération de Contenu par IA (Hackathon 2)
Présentation du projet

Ce projet expérimental propose deux volets complémentaires autour de la génération de contenu par IA, développés lors d’un hackathon :

    Volet 1 : Génération de posts (type LinkedIn) – Un pipeline modulaire pour générer automatiquement un texte court à partir d’un sujet donné, avec contrôles de qualité : résumé automatique, vérification de cohérence, filtrage éthique, et même une tentative de génération d’image via un VAE. L’objectif était d’obtenir un système 100% exécutable sur CPU (pas de GPU ni d’API externe) pour produire des posts de qualité tout en évitant les dérives (hallucinations, contenus offensants). Ce volet n’a pas été entièrement finalisé, notamment par manque de modèles puissants optimisés CPU et de temps pour peaufiner les résultats.

    Volet 2 : Génération d’articles de blog – Une seconde approche, fonctionnelle, axée sur la génération d’articles de blog (~300 mots) à partir d’un thème. Ce pipeline produit un article structuré (introduction, parties, conclusion), génère un résumé automatique de l’article, et illustre le tout avec une image créée par un modèle de diffusion. L’ensemble du processus est automatisé et peut être planifié (génération périodique d’articles), avec des scripts d’ordonnancement et une interface de visualisation des articles générés. Ce volet utilise des modèles plus récents et performants (toujours exécutables sur CPU) pour améliorer la cohérence et la qualité du contenu.

Volet 1 : Génération de posts (LinkedIn)

Objectif : Automatiser la création de posts courts de type LinkedIn à partir d’un prompt/sujet, tout en filtrant les contenus indésirables. Le pipeline complet enchaîne plusieurs étapes de NLP :

Prompt utilisateur 
↓
distilGPT2 — génération du texte initial 
↓
distilBART — résumé automatique du texte 
↓
MiniLM — vérification de similarité résumé/sujet 
↓
Filtre éthique (regex & mots-clés) 
↓
✅ Texte final accepté (ou ❌ rejeté si contenu inapproprié)

Détails du pipeline : L’utilisateur fournit un sujet de post. Un modèle GPT-2 distillé (distilgpt2) génère un texte initial relativement long. Ce texte est ensuite résumé via un modèle DistilBART-CNN (distilBART entraîné sur CNN/DailyMail) pour obtenir une version condensée plus cohérente (réduction des éventuelles « hallucinations » du modèle GPT-2). On calcule ensuite la similarité sémantique entre le sujet initial et le résumé à l’aide d’un encodeur Sentence-Transformers MiniLM (all-MiniLM-L6-v2) afin de s’assurer que le contenu généré reste pertinent par rapport au prompt. Un filtrage éthique est appliqué sur le texte (par exemple via des regex ou une liste de mots sensibles) pour détecter des propos offensants, du contenu inapproprié ou des dérives non souhaitées. Si le texte échoue aux critères (faible similarité ou contenu problématique), il peut être marqué ou rejeté ; sinon, le post final est produit. Le pipeline prévoit également un marquage du texte final (par exemple en ajoutant des indications si le contenu a été filtré ou modifié).

Évaluation : Des scripts d’évaluation (calcul du BLEU, ROUGE-L, etc.) permettent de mesurer la qualité des textes générés et la robustesse du pipeline. Un fichier de prompts adversariaux (adversarial_prompts.txt) est fourni pour tester la robustesse du système face à des entrées absurdes ou malveillantes. Lors des tests initiaux, le pipeline obtenait environ BLEU = 0,25, ROUGE-L = 0,45, une similarité moyennne de ~0,72 entre le résumé et le prompt, et un taux de détection éthique de ~5% (proportion de textes contenant des mots sensibles détectés).

Limites : Ce premier volet a démontré la faisabilité d’une génération multi-étapes sur CPU, mais avec des compromis de performance. Les modèles allégés (distilGPT2, distilBART, MiniLM) tournent sans GPU mais restent limités en qualité : le contenu généré peut manquer de richesse ou de fiabilité, nécessitant le résumé pour corriger les incohérences. Malgré les contrôles, GPT-2 peut produire du hors-sujet ou du texte peu naturel. Le filtrage éthique par mots-clés/regex reste sommaire (une amélioration envisagée était d’entraîner un classifieur de toxicité local). Enfin, la génération d’images par VAE (réseau auto-encodeur variationnel) est restée expérimentale : un petit VAE a été entraîné (ex. sur MNIST) pour générer des images basiques, sans rapport sémantique avec le post (plutôt une preuve de concept). Faute de modèles d’image adaptés au CPU et de données suffisantes, cette partie n’a pas abouti à des illustrations exploitables pour un post LinkedIn professionnel.
Volet 2 : Génération d’articles de blog

Objectif : Produire de façon autonome des articles de blog complets, avec résumé et image, à partir d’un thème donné. Ce second pipeline améliore la qualité du texte et intègre une véritable génération d’images. Le processus global est le suivant :

Sujet d’article (thème) 
↓
LaMini-Flan-T5 (783 M) — génération de l’article complet 
↓
DistilBART — résumé automatique de l’article 
↓
MiniLM — vérification de similarité article/résumé 
↓
Filtre éthique (profanités via better-profanity) 
↓
Stable Diffusion (Diffusers) — image d’illustration 
↓
Article final en Markdown (texte + résumé + image)

Détails du pipeline : On utilise un modèle de texte plus puissant, LaMini-Flan-T5 (783 millions de paramètres), capable de suivre des consignes en français. L’article est généré en une étape sous forme structurée : le prompt de génération inclut des instructions explicites (« Écris un article de blog structuré de 250 à 300 mots… avec Titre, introduction, parties, conclusion… ») afin d’obtenir un texte bien organisé et de longueur cible sans étape de résumé intermédiaire. Néanmoins, pour extraire un résumé court (par exemple pour un aperçu ou SEO), le pipeline emploie un DistilBART (identique au volet 1) ou un modèle équivalent pour résumer l’article généré. La cohérence entre l’article et son résumé est vérifiée avec le même Sentence-Transformer MiniLM, pour détecter d’éventuelles digressions. Le filtre éthique a été amélioré en intégrant la librairie better-profanity qui repère les insultes/propos offensants de manière plus systématique (via un dictionnaire de grossièretés).

Une fois le texte validé, le pipeline génère une image d’illustration du sujet grâce à Stable Diffusion (via HuggingFace Diffusers). Pour rester dans des temps de calcul raisonnables sur CPU, on utilise un nombre réduit d’itérations (par ex. 15 steps de diffusion) et un prompt d’image simplifié (par ex. « {sujet}, vector illustration, flat design, vibrant colors » avec un prompt négatif pour éviter le texte ou le flou). L’image obtenue (format PNG) illustre visuellement le thème de l’article.

Production et sortie : L’article final est sauvegardé dans un fichier Markdown comprenant : un titre (le sujet), le texte de l’article, puis une section “Résumé” générée, une indication du score de similarité sémantique, et le statut du filtrage éthique (✅ aucun problème ou ⚠️ détails des contenus sensibles détectés). Si une illustration a été créée, elle est référencée dans le Markdown. Un fichier JSON de métadonnées accompagne chaque article (contenant le titre/sujet, la date/heure de génération, le résumé, le score de similarité, le statut du filtre et les étiquettes de contenu sensible le cas échéant, ainsi que le nom de l’image associée).

Les articles et leurs ressources sont rangés par date dans un dossier de sortie (voir arborescence). Une interface Streamlit (blog_viewer.py) permet d’afficher la liste des articles générés avec leur date, le contenu formaté et l’illustration, simulant un blog automatisé alimenté par l’IA.

Ordonnancement : Ce volet intègre un script de planification (run_scheduler.py) utilisant la librairie schedule. Par exemple, il peut être configuré pour générer automatiquement un nouvel article toutes les heures ou tous les jours à heure fixe. (Le sujet peut être fixé dans le script ou choisi aléatoirement/parmi une liste de thèmes.) Cela permet d’automatiser la création de contenu régulier sur le blog IA.

Limites : Malgré l’utilisation de modèles plus grands, tout tourne sur CPU, ce qui implique des temps de génération assez longs (plusieurs dizaines de secondes pour générer ~300 mots, et jusqu’à 1-2 minutes pour l’image selon la machine). La qualité des articles est nettement meilleure qu’avec GPT-2, mais reste tributaire du modèle T5 utilisé : il peut encore y avoir des imprécisions ou un style générique. Le résumé automatique peut parfois répéter des infos évidentes. La génération d’images via Stable Diffusion sur CPU est très lente et consommatrice en mémoire ; la résolution et les détails ont été limités (ex: style illustration vectorielle) pour accélérer le rendu. Enfin, le pipeline n’est pas à l’abri de hallucinations ou de biais du modèle (même si on les a réduits) : une supervision humaine reste recommandée avant publication réelle des contenus.
Technologies et modèles utilisés

Les deux volets s’appuient sur un écosystème Python centré sur le NLP de HuggingFace et d’autres librairies open-source :

    Transformers (HuggingFace) – Utilisé pour charger et exécuter les modèles de génération de texte et de résumé.

        Modèles texte : distilgpt2 (GPT-2 distillé) pour la génération initiale (volet 1), puis MBZUAI/LaMini-Flan-T5-783M (un modèle Flan-T5 de 783M paramètres) pour la génération d’articles (volet 2).

        Modèle de résumé : sshleifer/distilbart-cnn-12-6 (DistilBART entraîné sur CNN/Daily Mail) pour résumer les textes longs.

        Autres : la librairie transformers est également utilisée pour le fine-tuning LoRA GPT-2 (une tentative d’adapter GPT-2-medium via Low-Rank Adaption, visible dans le dépôt, bien que cela ne soit pas intégré au pipeline final).

    Sentence-Transformers – Fournit le modèle all-MiniLM-L6-v2 (MiniLM) qui génère des embeddings sémantiques pour mesurer la similarité cosinus entre le prompt, le résumé et/ou l’article. Ceci permet un scoring de cohérence (vérification que le contenu généré reste dans le sujet).

    Diffusers (HuggingFace) – Pour la génération d’images par diffusion (Stable Diffusion). Un pipeline de diffusion textuelle est employé (modèle Stable Diffusion v1.5 par défaut) afin de créer une image correspondant au thème de l’article, en utilisant uniquement le CPU. Inclut l’utilisation de accelerate et safetensors pour optimiser le chargement des modèles de diffusion en mémoire.

    PyTorch – Backend principal pour les modèles de deep learning (transformers et diffusers). Utilisé également pour le VAE personnalisé (définition du modèle dans vae_model.py et entraînement sur un petit dataset).

    NLTK, rouge-score – Employés pour l’évaluation linguistique (tokenization et calcul de métriques BLEU/ROUGE dans le volet 1, via nltk pour BLEU et rouge-score pour ROUGE-L). NLTK sert aussi dans le volet 2 pour des utilitaires NLP (par ex. découpage en phrases si nécessaire).

    scikit-learn – Utilisé notamment pour des calculs de similarité (cosine similarity) ou d’autres métriques d’évaluation. Dans ce contexte, il peut intervenir pour la partie vérification de cohérence ou le traitement des vecteurs issus de Sentence-Transformers.

    better-profanity – Librairie légère de détection de grossièretés/profanités dans du texte. Intégrée dans ethical_filter_v2.py (volet 2) pour améliorer le filtrage éthique en repérant automatiquement les mots offensants courants (en français et anglais).

    schedule – Librairie de planification d’événements cron-like. Utilisée dans les scripts scheduler.py (volet 1) et run_scheduler.py (volet 2) pour automatiser l’exécution périodique du pipeline (par ex. toutes les heures). Cela permet de déployer le système en tâche de fond générant régulièrement du contenu.

    Streamlit – Framework web pour créer rapidement des interfaces utilisateur. Deux applications Streamlit sont fournies : app.py (volet 1) propose une interface de démo pour générer des posts filtrés en direct, et blog_viewer.py (volet 2) offre une interface type dashboard pour visualiser les articles de blog générés (avec leur résumé, stats et image). Ces UIs ont été utilisées pour la présentation du projet (elles ne sont pas obligatoires pour faire fonctionner les pipelines, mais facilitent l’interaction et la démonstration).

    Autres : datasets (HuggingFace Datasets) pour éventuellement charger un jeu de données (un exemple mentionné est IMDB, possiblement utilisé lors de tests ou d’un fine-tuning), peft (Parameter-Efficient Fine-Tuning) pour le LoRA GPT-2 fine-tuning, torchvision et matplotlib pour le VAE (traitement d’images, affichage de résultats d’entraînement), PyPDF2, pdfplumber et python-docx pour la lecture de documents (PDF/DOCX) – ces derniers étaient prévus pour enrichir le système (par ex. extraire du texte de documents d’entrée), mais sont optionnels et non au cœur de la génération de contenu.

Arborescence du projet

Le répertoire ai_generation/ contient l’ensemble des fichiers source, organisés en parties correspondantes aux deux volets :

ai_generation/  
├── README.md                  - Présentation initiale du projet (pipeline du volet LinkedIn)  
├── adversarial_prompts.txt    - Prompts « adversaires » pour tester la robustesse (volet 1)  
├── app.py                     - Application web Streamlit de démonstration (volet 1)  
├── automate.py                - Script CLI pour exécuter le pipeline complet une fois (volet 1)  
├── scheduler.py               - Script d’ordonnancement (schedule) pour exécutions régulières (volet 1)  
├── data_loader.py             - Chargement de dataset (ex: IMDB) pour tests ou entraînement  
├── ethical_filter.py          - Filtrage éthique basique par mots-clés/expressions régulières  
├── evaluation.py              - Évaluation du texte généré (métriques BLEU, ROUGE, etc., volet 1)  
├── generation.py              - Générateur de texte (volet 1, classe *TextGenerator* avec distilGPT2)  
├── summarization.py           - Résumeur de texte (volet 1, utilise DistilBART CNN)  
├── similarity.py              - Vérification de similarité sémantique (encodeur MiniLM)  
├── utils.py                   - Fonctions utilitaires diverses (volet 1)  
├── vae_images/                - **(Expérimental)** Génération d’images par VAE  
│   ├── vae_model.py           - Définition du modèle VAE (auto-encodeur variationnel)  
│   ├── train_vae.py           - Entraîne le VAE (exemple sur dataset MNIST)  
│   └── generate_image.py      - Génère une image à partir du VAE entraîné  
├── gpt2_lora_finetuned/       - **(Expérimental)** Modèle GPT-2 fine-tuné avec LoRA (poids adaptatifs)  
│   ├── adapter_config.json    - Configuration LoRA du modèle  
│   ├── adapter_model.bin      - Poids LoRA entraînés  
│   └── checkpoint-2/…         - Données de checkpoint (issu de l’entraînement LoRA)  
├── new_test/                  - **Volet 2 :** pipeline de génération d’articles de blog  
│   ├── article_generator.py   - Générateur d’article de blog (classe *ArticleGenerator*, modèle LaMini-Flan-T5)  
│   ├── summarization.py       - Résumeur de texte (classe *TextSummarizer*, réutilise DistilBART)  
│   ├── similarity.py          - Vérificateur de similarité (classe *SimilarityChecker*, MiniLM)  
│   ├── ethical_filter_v2.py   - Filtrage éthique v2 (fonction `ethical_filter` utilisant better-profanity)  
│   ├── image_gen.py           - Générateur d’image (classe *ImageGenerator* utilisant Diffusers Stable Diffusion)  
│   ├── pipeline.py            - Pipeline complet : enchaîne génération article, résumé, image, sauvegarde  
│   ├── run_scheduler.py       - Ordonnancement planifié (génère des articles à intervalles réguliers)  
│   ├── blog_viewer.py         - Application Streamlit pour visualiser les articles générés (type blog)  
│   ├── simple_app.py          - Application de démonstration simplifiée (optionnelle, ex. génération one-shot)  
│   ├── requirements.txt       - Requirements spécifiques du volet 2 (versions de lib mises à jour)  
│   └── outputs/               - Dossiers de sortie des articles de blog générés  
│       └── YYYYMMDD_HHMMSS/   - Un dossier par article généré (horodaté)  
│           ├── article.md     - Article en Markdown (texte + résumé + stats + image intégrée)  
│           └── meta.json      - Métadonnées JSON (titre, date, résumé, similarité, éthique, image…)  
└── requirements.txt           - Requirements globaux du projet (volet 1 et dépendances communes)  

👉 Remarque : Le dossier new_test/ contient la version aboutie du pipeline (volet 2). Les fichiers du niveau supérieur (hors new_test/) correspondent en grande partie au premier prototype (volet 1) et ne sont pas tous utilisés dans le volet 2. On retrouve par exemple deux versions de ethical_filter (simple et v2), de même que certaines fonctionnalités expérimentales (vae_images, fine-tuning GPT-2) présentes pour historique mais non actives dans la version finale du blog.
Installation et utilisation

    Prerequis : Assurez-vous d’avoir Python 3.10+ installé. Ces pipelines étant conçus pour fonctionner sur CPU, aucune configuration GPU n’est requise, mais cela implique des temps de calcul plus longs. 

1. Récupération du projet – Clonez le dépôt GitHub ou téléchargez le dossier ai_generation/ et placez-vous dedans :

git clone https://github.com/Pistou27/hackathon2.git
cd hackathon2/ai_generation

2. Installation des dépendances – Deux fichiers de requirements sont fournis :

    Pour le volet 1 (posts LinkedIn), vous pouvez installer les dépendances d’origine :

pip install -r requirements.txt

Cela inclut Transformers 4.41, Torch 2.3, etc., suffisant pour exécuter le pipeline initial.

Pour le volet 2 (blog), des versions plus récentes de certaines librairies sont nécessaires (Transformers 4.54, Torch 2.7.1, Diffusers 0.28, etc.). Il est conseillé d’installer les requirements spécifiques :

    pip install -r new_test/requirements.txt

    (Selon votre environnement, ces installations peuvent être faites dans un virtualenv distinct pour éviter les conflits de versions. Sinon, installer directement le second requirements.txt couvrira la plupart des besoins du volet 1, car il contient des versions supérieures ou égales.)

3. Utilisation du volet 1 (pipeline posts LinkedIn) – Ce prototype peut être testé de plusieurs façons :

    Ligne de commande (pipeline complet) : exécutez python automate.py. Le script générera une série de posts de test (par exemple 5 prompts prédéfinis dans main.py ou automate.py) et affichera dans la console le texte généré, éventuellement annoté des scores/indicateurs (similarité, filtrage...).

    Interface Streamlit : lancez streamlit run app.py. Une interface web locale s’ouvrira, permettant de saisir un sujet de post et d’obtenir en direct le texte généré filtré. (La page Streamlit configure une présentation simple avec un thème de couleur, et affiche le résultat du pipeline étape par étape dans l’application web).

    Planification automatique : le script scheduler.py (aussi référencé comme automate_schedule dans la documentation) devait permettre d’exécuter automate.py à intervalle régulier. Vous pouvez l’adapter et l’utiliser en faisant python scheduler.py (par défaut, il peut être configuré pour un certain timing – veillez à ajuster l’intervalle voulu dans le code). Ce script démarrera un scheduler qui tourne en tâche de fond et génère périodiquement de nouveaux contenus. Note : Ce volet était un prototype, il se peut que la planification ou d’autres aspects nécessitent des ajustements manuels dans le code pour fonctionner parfaitement.

4. Utilisation du volet 2 (pipeline blog) – C’est la partie recommandée pour une démonstration aboutie du projet.

    Génération d’un article : exécutez le script principal du pipeline avec un sujet en argument. Par exemple :

cd new_test
python pipeline.py "Les avancées de l'IA en 2025"

Ceci lancera la génération d’un article de blog sur le thème "Les avancées de l'IA en 2025". Par défaut, le pipeline crée aussi une image d’illustration. Le résultat complet sera sauvegardé dans new_test/outputs/<timestamp>/ (le nom du dossier est la date/heure courante). Vous y trouverez le article.md (que vous pouvez ouvrir pour lire le contenu formaté Markdown) et le illustration.png généré, ainsi que meta.json. La console affichera le déroulement (étapes de génération, scores...) et le chemin du dossier résultat.
Options: vous pouvez désactiver la génération d’image en ajoutant --no-image à la commande. Vous pouvez aussi spécifier un répertoire de sortie différent via --out_dir <path>. Si aucun sujet n’est fourni en argument, le script utilisera un sujet par défaut (défini dans le code, ex: "Les avancées récentes de l'IA").

Visualisation des articles : pour parcourir et lire aisément les articles générés, lancez l’interface Streamlit dédiée :

    streamlit run new_test/blog_viewer.py

    Dans votre navigateur, vous verrez un tableau de bord listant les articles (dossiers dans outputs/) par ordre décroissant. Pour chaque article, le titre, la date de génération, le score de similarité et les labels du filtre éthique sont affichés, suivis du contenu formaté (titre, texte, résumé) et de l’image. Cela simule un blog alimenté automatiquement.

    Génération planifiée automatique : utilisez le script new_test/run_scheduler.py pour lancer une génération périodique. Par exemple, par défaut il peut être codé pour créer un article chaque heure pile. Vous pouvez l’exécuter avec python new_test/run_scheduler.py. Le script utilise la bibliothèque schedule pour programmer la fonction de génération (job()) à la fréquence désirée (ici toutes les heures, voir le code pour ajuster l’intervalle ou le sujet). Laissez-le tourner en tâche de fond : il déclenchera pipeline.py à chaque intervalle. (Par défaut dans le code fourni, le sujet est fixé à "Les avancées de l'IA en 2025" pour toutes les itérations – vous pouvez modifier la variable topic dans job() pour varier les thèmes ou en sélectionner aléatoirement).

5. Extensibilité – Vous pouvez adapter ces pipelines à vos besoins. Par exemple, changer le modèle de génération (en choisissant un autre modèle Transformers plus grand pour de meilleurs résultats – en gardant à l’esprit les contraintes CPU), enrichir le filtre éthique (intégrer un classifieur de toxicité entraîné en local, comme envisagé en amélioration), ou modifier la fréquence/les sujets de génération planifiée. L’architecture modulaire (classes séparées pour génération de texte, résumé, image, etc.) facilite les remplacements de composants. De même, l’interface Streamlit peut être customisée pour une meilleure présentation ou pour permettre de déclencher manuellement la génération d’un nouvel article à la demande.
Limitations et performances (CPU)

    Tout le projet a été développé avec la contrainte CPU uniquement (pas de GPU ni cloud payant). Cela influence les temps de traitement : générer un article avec image peut prendre de 30 secondes à quelques minutes selon votre machine. Il est recommandé d’utiliser un CPU puissant, et éventuellement de réduire certaines charges (par ex. désactiver l’image si non nécessaire, ou utiliser un modèle de diffusion plus léger si disponible).

    Les modèles distillés (distilGPT2, DistilBART) ont été choisis pour leur légèreté, mais ils produisent un contenu moins riche/moins fiable que des modèles plus grands. Le compromis entre performance et qualité est notable. Dans le volet 2, le modèle Flan-T5 783M offre une meilleure qualité de texte que GPT-2 distillé, au prix d’un temps de calcul un peu supérieur.

    La génération d’images via Stable Diffusion sur CPU est particulièrement lourde. Le pipeline utilise 15 étapes de diffusion (au lieu de 50 ou 100 habituellement) et une résolution implicite limitée, ce qui réduit la qualité/détail de l’image pour gagner du temps. Malgré cela, produire une image reste l’étape la plus lente. Si le temps ou les ressources sont un problème, on peut envisager de désactiver l’image ou d’utiliser une VAE pré-entrainé plus simple comme dans le volet 1 (mais les images seront alors très basiques).

    Ce projet étant un projet de hackathon, tout n’est pas entièrement abouti ou optimisé. Des bugs mineurs peuvent subsister, et certaines parties expérimentales du code ne s’intègrent pas dans le pipeline principal (par ex. le fine-tuning LoRA ou la lecture PDF ne sont pas exploités dans le flux final). Néanmoins, le second volet a été testé de bout en bout et doit fournir un résultat cohérent.

Auteurs et licence

Ce projet a été réalisé dans le cadre d’un hackathon par Jérémy Novico et Alexandre Perrault (GitHub : Jeynova et aperrault27), avec le support du profil GitHub Pistou27. Le dépôt ne spécifie pas de licence open-source explicite. Par conséquent, par défaut le code est non libre de droits (tous droits réservés aux auteurs). Si vous envisagez de réutiliser ou de diffuser ce code, il est conseillé de contacter les auteurs pour accord.

© 2025 - Projet Hackathon2 AI Generation. Ce document README.md est une reconstitution détaillée du projet à partir des sources disponibles, destinée à expliquer le fonctionnement et la structure du système.