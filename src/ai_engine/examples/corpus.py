"""Curated corpus of gold-standard game examples and educational resources.

Each entry is a plain dict that will be converted to a
:class:`~ai_engine.rag.document.Document` by the injector.

Two document families:

* **game_example** – A complete, valid JSON game object.  Tagged with
  ``game_type``, ``language``, and ``category`` so the retriever surfaces
  them as few-shot references when a matching generation is requested.

* **educational_resource** – Factual paragraphs about a category/topic
  that the LLM can cite instead of fabricating information.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────


def _example_doc(
    game_type: str,
    language: str,
    category: str,
    game_obj: dict[str, Any],
) -> dict[str, Any]:
    """Build a game_example document dict."""
    formatted = json.dumps(game_obj, ensure_ascii=False, indent=2)
    content = (
        f"[EXAMPLE — {game_type} | {language} | {category}]\n"
        f"The following is a complete, valid {game_type} game object "
        f"in {language} about «{category}».\n\n"
        f"```json\n{formatted}\n```"
    )
    return {
        "content": content,
        "doc_id": f"example-{game_type}-{language}-{category.lower().replace(' ', '-')[:40]}",
        "metadata": {
            "kind": "game_example",
            "game_type": game_type,
            "language": language,
            "category": category,
        },
    }


def _resource_doc(
    category: str,
    language: str,
    text: str,
    *,
    sub_topic: str = "",
) -> dict[str, Any]:
    """Build an educational_resource document dict."""
    slug = category.lower().replace(" ", "-")[:40]
    sub = f"-{sub_topic.lower().replace(' ', '-')[:30]}" if sub_topic else ""
    return {
        "content": text.strip(),
        "doc_id": f"resource-{slug}-{language}{sub}",
        "metadata": {
            "kind": "educational_resource",
            "category": category,
            "language": language,
            "sub_topic": sub_topic,
        },
    }


# =====================================================================
#  QUIZ EXAMPLES
# =====================================================================

QUIZ_EXAMPLES: list[dict[str, Any]] = [
    # ── General Knowledge · ES ────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "General Knowledge",
        {
            "game_type": "quiz",
            "title": "Cultura General — Ronda Básica",
            "difficulty_percentage": 40,
            "questions": [
                {
                    "question": "¿Cuál es el océano más grande del mundo?",
                    "options": ["Atlántico", "Índico", "Pacífico", "Ártico"],
                    "correct_index": 2,
                    "explanation": "El océano Pacífico cubre aproximadamente 165 millones de km², siendo el más extenso del planeta.",
                },
                {
                    "question": "¿En qué año llegó el ser humano a la Luna por primera vez?",
                    "options": ["1965", "1969", "1971", "1973"],
                    "correct_index": 1,
                    "explanation": "El 20 de julio de 1969, la misión Apollo 11 de la NASA logró el primer alunizaje tripulado.",
                },
                {
                    "question": "¿Cuántos huesos tiene el cuerpo humano adulto?",
                    "options": ["186", "206", "226", "256"],
                    "correct_index": 1,
                    "explanation": "Un adulto posee 206 huesos; los bebés nacen con alrededor de 270 que se fusionan con el tiempo.",
                },
                {
                    "question": "¿Cuál es el planeta más cercano al Sol?",
                    "options": ["Venus", "Marte", "Mercurio", "Tierra"],
                    "correct_index": 2,
                    "explanation": "Mercurio orbita a una distancia media de 57,9 millones de km del Sol.",
                },
                {
                    "question": "¿Qué gas es esencial para la respiración humana?",
                    "options": ["Nitrógeno", "Dióxido de carbono", "Oxígeno", "Helio"],
                    "correct_index": 2,
                    "explanation": "El oxígeno (O₂) constituye alrededor del 21 % de la atmósfera y es imprescindible para la respiración celular.",
                },
            ],
        },
    ),
    # ── General Knowledge · EN ────────────────────────────────────
    _example_doc(
        "quiz",
        "en",
        "General Knowledge",
        {
            "game_type": "quiz",
            "title": "General Knowledge — Starter Round",
            "difficulty_percentage": 40,
            "questions": [
                {
                    "question": "What is the largest ocean on Earth?",
                    "options": ["Atlantic", "Indian", "Pacific", "Arctic"],
                    "correct_index": 2,
                    "explanation": "The Pacific Ocean spans roughly 165 million km², making it the largest by area.",
                },
                {
                    "question": "In which year did humans first land on the Moon?",
                    "options": ["1965", "1969", "1971", "1973"],
                    "correct_index": 1,
                    "explanation": "NASA's Apollo 11 mission achieved the first crewed Moon landing on 20 July 1969.",
                },
                {
                    "question": "How many bones does the adult human body have?",
                    "options": ["186", "206", "226", "256"],
                    "correct_index": 1,
                    "explanation": "Adults have 206 bones; infants are born with about 270 that fuse over time.",
                },
                {
                    "question": "Which planet is closest to the Sun?",
                    "options": ["Venus", "Mars", "Mercury", "Earth"],
                    "correct_index": 2,
                    "explanation": "Mercury orbits at an average distance of 57.9 million km from the Sun.",
                },
                {
                    "question": "What gas is essential for human respiration?",
                    "options": ["Nitrogen", "Carbon dioxide", "Oxygen", "Helium"],
                    "correct_index": 2,
                    "explanation": "Oxygen (O₂) makes up about 21 % of the atmosphere and is vital for cellular respiration.",
                },
            ],
        },
    ),
    # ── Science & Nature · ES ─────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "Science & Nature",
        {
            "game_type": "quiz",
            "title": "Ciencia y Naturaleza — Nivel Intermedio",
            "difficulty_percentage": 55,
            "questions": [
                {
                    "question": "¿Cuál es el símbolo químico del oro?",
                    "options": ["Ag", "Au", "Fe", "Cu"],
                    "correct_index": 1,
                    "explanation": "El símbolo Au proviene del latín 'aurum'. La plata es Ag y el hierro Fe.",
                },
                {
                    "question": "¿Qué tipo de animal es una salamandra?",
                    "options": ["Reptil", "Anfibio", "Mamífero", "Pez"],
                    "correct_index": 1,
                    "explanation": "Las salamandras son anfibios del orden Urodela, con piel húmeda y sin escamas.",
                },
                {
                    "question": "¿Cuántos cromosomas tiene una célula humana?",
                    "options": ["23", "44", "46", "48"],
                    "correct_index": 2,
                    "explanation": "Las células somáticas humanas contienen 46 cromosomas (23 pares).",
                },
                {
                    "question": "¿Qué fuerza mantiene a los planetas en órbita alrededor del Sol?",
                    "options": [
                        "Electromagnética",
                        "Nuclear fuerte",
                        "Gravitatoria",
                        "Nuclear débil",
                    ],
                    "correct_index": 2,
                    "explanation": "La fuerza gravitatoria, descrita por Newton, es responsable de las órbitas planetarias.",
                },
                {
                    "question": "¿Cuál es el hueso más largo del cuerpo humano?",
                    "options": ["Tibia", "Húmero", "Fémur", "Radio"],
                    "correct_index": 2,
                    "explanation": "El fémur, ubicado en el muslo, mide aproximadamente una cuarta parte de la estatura total.",
                },
            ],
        },
    ),
    # ── History · ES ──────────────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "History",
        {
            "game_type": "quiz",
            "title": "Historia Universal — Hechos Clave",
            "difficulty_percentage": 50,
            "questions": [
                {
                    "question": "¿En qué año cayó el Muro de Berlín?",
                    "options": ["1987", "1989", "1991", "1993"],
                    "correct_index": 1,
                    "explanation": "El Muro de Berlín cayó el 9 de noviembre de 1989, marcando el fin de la Guerra Fría.",
                },
                {
                    "question": "¿Quién fue el primer emperador romano?",
                    "options": ["Julio César", "Nerón", "Augusto", "Trajano"],
                    "correct_index": 2,
                    "explanation": "Augusto (Octavio) se convirtió en el primer emperador de Roma en el 27 a.C.",
                },
                {
                    "question": "¿En qué continente se originó la Revolución Industrial?",
                    "options": ["América", "Asia", "Europa", "África"],
                    "correct_index": 2,
                    "explanation": "La Revolución Industrial comenzó en Gran Bretaña a mediados del siglo XVIII.",
                },
                {
                    "question": "¿Qué civilización construyó Machu Picchu?",
                    "options": ["Maya", "Azteca", "Inca", "Olmeca"],
                    "correct_index": 2,
                    "explanation": "Machu Picchu fue construido por los incas en el siglo XV en los Andes peruanos.",
                },
                {
                    "question": "¿En qué año comenzó la Primera Guerra Mundial?",
                    "options": ["1912", "1914", "1916", "1918"],
                    "correct_index": 1,
                    "explanation": "La Primera Guerra Mundial estalló en 1914 tras el asesinato del archiduque Francisco Fernando.",
                },
            ],
        },
    ),
    # ── Entertainment: Film · EN ──────────────────────────────────
    _example_doc(
        "quiz",
        "en",
        "Entertainment: Film",
        {
            "game_type": "quiz",
            "title": "Movies & Cinema — Pop Culture",
            "difficulty_percentage": 45,
            "questions": [
                {
                    "question": "Who directed the 1994 film 'Pulp Fiction'?",
                    "options": [
                        "Martin Scorsese",
                        "Quentin Tarantino",
                        "Steven Spielberg",
                        "David Fincher",
                    ],
                    "correct_index": 1,
                    "explanation": "Quentin Tarantino wrote and directed Pulp Fiction, which won the Palme d'Or at Cannes.",
                },
                {
                    "question": "Which film won the first Academy Award for Best Picture?",
                    "options": [
                        "Wings",
                        "Sunrise",
                        "The Jazz Singer",
                        "All Quiet on the Western Front",
                    ],
                    "correct_index": 0,
                    "explanation": "Wings (1927) won Best Picture at the 1st Academy Awards ceremony in 1929.",
                },
                {
                    "question": "In 'The Matrix', what colour pill does Neo take?",
                    "options": ["Blue", "Green", "Red", "Yellow"],
                    "correct_index": 2,
                    "explanation": "Neo chooses the red pill, which reveals the true nature of the Matrix.",
                },
                {
                    "question": "Which studio produced 'Toy Story' (1995)?",
                    "options": ["DreamWorks", "Pixar", "Blue Sky", "Illumination"],
                    "correct_index": 1,
                    "explanation": "Toy Story was Pixar's first feature film the first entirely computer-animated feature.",
                },
                {
                    "question": "What fictional country is Black Panther from?",
                    "options": ["Latveria", "Wakanda", "Genosha", "Zamunda"],
                    "correct_index": 1,
                    "explanation": "Wakanda is a fictional African nation in the Marvel Cinematic Universe rich in vibranium.",
                },
            ],
        },
    ),
    # ── Sports · ES ───────────────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "Sports",
        {
            "game_type": "quiz",
            "title": "Deportes — Conocimientos Generales",
            "difficulty_percentage": 45,
            "questions": [
                {
                    "question": "¿Cuántos jugadores tiene un equipo de fútbol en el campo?",
                    "options": ["9", "10", "11", "12"],
                    "correct_index": 2,
                    "explanation": "Cada equipo de fútbol tiene 11 jugadores en el campo, incluyendo al portero.",
                },
                {
                    "question": "¿En qué país se celebraron los Juegos Olímpicos de 2020?",
                    "options": ["China", "Japón", "Corea del Sur", "Brasil"],
                    "correct_index": 1,
                    "explanation": "Los Juegos Olímpicos de Tokio 2020 se celebraron en Japón en 2021 por la pandemia.",
                },
                {
                    "question": "¿Cuántos sets necesita ganar un tenista para ganar un partido de Grand Slam masculino?",
                    "options": ["2", "3", "4", "5"],
                    "correct_index": 1,
                    "explanation": "En Grand Slam masculino se juega al mejor de 5 sets, necesitando ganar 3.",
                },
                {
                    "question": "¿Qué deporte se practica en el Tour de Francia?",
                    "options": ["Atletismo", "Ciclismo", "Natación", "Automovilismo"],
                    "correct_index": 1,
                    "explanation": "El Tour de Francia es la carrera ciclista más prestigiosa del mundo, creada en 1903.",
                },
                {
                    "question": "¿Cuál es la distancia de una maratón?",
                    "options": ["21,1 km", "42,195 km", "50 km", "100 km"],
                    "correct_index": 1,
                    "explanation": "La maratón mide exactamente 42,195 km, distancia estandarizada desde 1921.",
                },
            ],
        },
    ),
    # ── Art · ES ──────────────────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "Art",
        {
            "game_type": "quiz",
            "title": "Arte — Grandes Obras y Artistas",
            "difficulty_percentage": 50,
            "questions": [
                {
                    "question": "¿Quién pintó 'La noche estrellada'?",
                    "options": [
                        "Claude Monet",
                        "Pablo Picasso",
                        "Vincent van Gogh",
                        "Salvador Dalí",
                    ],
                    "correct_index": 2,
                    "explanation": "Van Gogh pintó La noche estrellada en 1889 durante su estancia en el asilo de Saint-Rémy.",
                },
                {
                    "question": "¿En qué museo se encuentra la Mona Lisa?",
                    "options": [
                        "Museo del Prado",
                        "Museo del Louvre",
                        "Galería Uffizi",
                        "Museo Británico",
                    ],
                    "correct_index": 1,
                    "explanation": "La Gioconda de Leonardo da Vinci se exhibe en el Museo del Louvre de París desde 1797.",
                },
                {
                    "question": "¿Qué estilo artístico lideró Salvador Dalí?",
                    "options": [
                        "Impresionismo",
                        "Cubismo",
                        "Surrealismo",
                        "Expresionismo",
                    ],
                    "correct_index": 2,
                    "explanation": "Dalí fue uno de los máximos representantes del surrealismo, junto a Magritte y Ernst.",
                },
                {
                    "question": "¿Quién esculpió el David de mármol expuesto en Florencia?",
                    "options": ["Donatello", "Bernini", "Miguel Ángel", "Rodin"],
                    "correct_index": 2,
                    "explanation": "Miguel Ángel esculpió el David entre 1501 y 1504; mide 5,17 metros de altura.",
                },
                {
                    "question": "¿Qué movimiento artístico nació con Monet y la obra 'Impresión, sol naciente'?",
                    "options": ["Realismo", "Barroco", "Impresionismo", "Romanticismo"],
                    "correct_index": 2,
                    "explanation": "El impresionismo tomó su nombre del cuadro de Monet exhibido en 1874.",
                },
            ],
        },
    ),
    # ── Geography · ES ────────────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "Geography",
        {
            "game_type": "quiz",
            "title": "Geografía — Capitales y Continentes",
            "difficulty_percentage": 40,
            "questions": [
                {
                    "question": "¿Cuál es la capital de Australia?",
                    "options": ["Sídney", "Melbourne", "Canberra", "Brisbane"],
                    "correct_index": 2,
                    "explanation": "Canberra es la capital federal de Australia desde 1913, elegida como compromiso entre Sídney y Melbourne.",
                },
                {
                    "question": "¿Cuál es el río más largo del mundo?",
                    "options": ["Amazonas", "Nilo", "Yangtsé", "Misisipi"],
                    "correct_index": 1,
                    "explanation": "El Nilo mide aproximadamente 6.650 km, aunque algunas mediciones recientes sugieren que el Amazonas podría superarlo.",
                },
                {
                    "question": "¿En qué continente se encuentra el desierto del Sahara?",
                    "options": ["Asia", "América", "África", "Oceanía"],
                    "correct_index": 2,
                    "explanation": "El Sahara ocupa gran parte del norte de África, con una superficie de unos 9 millones de km².",
                },
                {
                    "question": "¿Cuál es el país más grande del mundo por superficie?",
                    "options": ["China", "Canadá", "Estados Unidos", "Rusia"],
                    "correct_index": 3,
                    "explanation": "Rusia tiene una superficie de 17,1 millones de km², abarcando dos continentes.",
                },
                {
                    "question": "¿Cuál es la montaña más alta del mundo?",
                    "options": ["K2", "Kangchenjunga", "Monte Everest", "Lhotse"],
                    "correct_index": 2,
                    "explanation": "El Monte Everest mide 8.849 metros según la medición de 2020.",
                },
            ],
        },
    ),
    # ── Animals · ES ──────────────────────────────────────────────
    _example_doc(
        "quiz",
        "es",
        "Animals",
        {
            "game_type": "quiz",
            "title": "Animales — Curiosidades del Reino Animal",
            "difficulty_percentage": 35,
            "questions": [
                {
                    "question": "¿Cuál es el animal terrestre más rápido?",
                    "options": ["León", "Guepardo", "Gacela", "Caballo"],
                    "correct_index": 1,
                    "explanation": "El guepardo puede alcanzar velocidades de hasta 112 km/h en cortas distancias.",
                },
                {
                    "question": "¿Cuántas patas tiene una araña?",
                    "options": ["6", "8", "10", "12"],
                    "correct_index": 1,
                    "explanation": "Las arañas son arácnidos y poseen 8 patas, a diferencia de los insectos que tienen 6.",
                },
                {
                    "question": "¿Cuál es el mamífero más grande del planeta?",
                    "options": [
                        "Elefante africano",
                        "Ballena azul",
                        "Jirafa",
                        "Tiburón ballena",
                    ],
                    "correct_index": 1,
                    "explanation": "La ballena azul puede medir hasta 30 metros y pesar 180 toneladas.",
                },
                {
                    "question": "¿Qué animal es conocido por cambiar de color para camuflarse?",
                    "options": ["Lagartija", "Camaleón", "Iguana", "Gecko"],
                    "correct_index": 1,
                    "explanation": "Los camaleones cambian de color mediante células cromatóforas, aunque también lo hacen por temperatura y estado de ánimo.",
                },
                {
                    "question": "¿De qué se alimentan principalmente los koalas?",
                    "options": ["Bambú", "Eucalipto", "Hierba", "Insectos"],
                    "correct_index": 1,
                    "explanation": "Los koalas se alimentan casi exclusivamente de hojas de eucalipto, consumiendo hasta 500 g diarios.",
                },
            ],
        },
    ),
]


# =====================================================================
#  WORD-PASS EXAMPLES
# =====================================================================

WORD_PASS_EXAMPLES: list[dict[str, Any]] = [
    # ── General Knowledge · ES ────────────────────────────────────
    _example_doc(
        "word-pass",
        "es",
        "General Knowledge",
        {
            "game_type": "word-pass",
            "title": "Pasapalabra — Cultura General",
            "difficulty_percentage": 45,
            "words": [
                {
                    "letter": "A",
                    "hint": "Sexto continente descubierto, cubierto de hielo",
                    "answer": "Antártida",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Instrumento de viento metal con pistones",
                    "answer": "Bombardino",
                    "starts_with": True,
                },
                {
                    "letter": "C",
                    "hint": "Capital de Francia",
                    "answer": "Ciudad de París",
                    "starts_with": True,
                },
                {
                    "letter": "D",
                    "hint": "Deporte que se practica en una piscina",
                    "answer": "Natación",
                    "starts_with": False,
                },
                {
                    "letter": "E",
                    "hint": "País sudamericano cuya capital es Quito",
                    "answer": "Ecuador",
                    "starts_with": True,
                },
                {
                    "letter": "F",
                    "hint": "Proceso por el que las plantas elaboran su alimento",
                    "answer": "Fotosíntesis",
                    "starts_with": True,
                },
                {
                    "letter": "G",
                    "hint": "Fuerza que atrae los cuerpos hacia el centro de la Tierra",
                    "answer": "Gravedad",
                    "starts_with": True,
                },
                {
                    "letter": "H",
                    "hint": "Elemento químico más abundante del universo",
                    "answer": "Hidrógeno",
                    "starts_with": True,
                },
                {
                    "letter": "I",
                    "hint": "Isla más grande de Europa",
                    "answer": "Islandia",
                    "starts_with": True,
                },
                {
                    "letter": "J",
                    "hint": "Deporte de combate olímpico de origen japonés",
                    "answer": "Judo",
                    "starts_with": True,
                },
                {
                    "letter": "L",
                    "hint": "Satélite natural de la Tierra",
                    "answer": "Luna",
                    "starts_with": True,
                },
                {
                    "letter": "M",
                    "hint": "Unidad de longitud equivalente a mil metros",
                    "answer": "Kilómetro",
                    "starts_with": False,
                },
                {
                    "letter": "N",
                    "hint": "Gas que compone el 78 % de la atmósfera terrestre",
                    "answer": "Nitrógeno",
                    "starts_with": True,
                },
                {
                    "letter": "O",
                    "hint": "Cuerpo celeste que gira alrededor de una estrella",
                    "answer": "Órbita",
                    "starts_with": True,
                },
                {
                    "letter": "P",
                    "hint": "Profesional que estudia y trata la mente humana",
                    "answer": "Psicólogo",
                    "starts_with": True,
                },
                {
                    "letter": "R",
                    "hint": "Estación del año entre el invierno y el verano",
                    "answer": "Primavera",
                    "starts_with": False,
                },
                {
                    "letter": "S",
                    "hint": "Estrella más cercana a la Tierra",
                    "answer": "Sol",
                    "starts_with": True,
                },
                {
                    "letter": "T",
                    "hint": "Instrumento para medir la temperatura",
                    "answer": "Termómetro",
                    "starts_with": True,
                },
                {
                    "letter": "V",
                    "hint": "Planeta conocido como el lucero del alba",
                    "answer": "Venus",
                    "starts_with": True,
                },
                {
                    "letter": "Z",
                    "hint": "Animal equino con rayas blancas y negras",
                    "answer": "Zebra",
                    "starts_with": True,
                },
            ],
        },
    ),
    # ── General Knowledge · EN ────────────────────────────────────
    _example_doc(
        "word-pass",
        "en",
        "General Knowledge",
        {
            "game_type": "word-pass",
            "title": "Word Wheel — General Knowledge",
            "difficulty_percentage": 45,
            "words": [
                {
                    "letter": "A",
                    "hint": "The continent covered in ice at the South Pole",
                    "answer": "Antarctica",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "System of governance where citizens vote for representatives",
                    "answer": "Ballot",
                    "starts_with": True,
                },
                {
                    "letter": "C",
                    "hint": "Capital city of Australia",
                    "answer": "Canberra",
                    "starts_with": True,
                },
                {
                    "letter": "D",
                    "hint": "Planet known for its dramatic ring system",
                    "answer": "Saturn",
                    "starts_with": False,
                },
                {
                    "letter": "E",
                    "hint": "The line around the middle of the Earth at 0° latitude",
                    "answer": "Equator",
                    "starts_with": True,
                },
                {
                    "letter": "F",
                    "hint": "The study of fossils and ancient life",
                    "answer": "Fossil",
                    "starts_with": True,
                },
                {
                    "letter": "G",
                    "hint": "Force of attraction between two masses",
                    "answer": "Gravity",
                    "starts_with": True,
                },
                {
                    "letter": "H",
                    "hint": "Chemical element with symbol H and atomic number 1",
                    "answer": "Hydrogen",
                    "starts_with": True,
                },
                {
                    "letter": "I",
                    "hint": "Large island nation in the North Atlantic near the Arctic",
                    "answer": "Iceland",
                    "starts_with": True,
                },
                {
                    "letter": "J",
                    "hint": "Planet famous for its Great Red Spot",
                    "answer": "Jupiter",
                    "starts_with": True,
                },
                {
                    "letter": "L",
                    "hint": "Natural satellite orbiting the Earth",
                    "answer": "Luna",
                    "starts_with": True,
                },
                {
                    "letter": "M",
                    "hint": "Red planet and fourth from the Sun",
                    "answer": "Mars",
                    "starts_with": True,
                },
                {
                    "letter": "N",
                    "hint": "Gas making up 78 % of Earth's atmosphere",
                    "answer": "Nitrogen",
                    "starts_with": True,
                },
                {
                    "letter": "O",
                    "hint": "Gas essential for human respiration, about 21% of air",
                    "answer": "Oxygen",
                    "starts_with": True,
                },
                {
                    "letter": "P",
                    "hint": "The study of the mind and behaviour",
                    "answer": "Psychology",
                    "starts_with": True,
                },
                {
                    "letter": "R",
                    "hint": "Electromagnetic radiation we perceive as colours",
                    "answer": "Rainbow",
                    "starts_with": True,
                },
                {
                    "letter": "S",
                    "hint": "The closest star to Earth",
                    "answer": "Sun",
                    "starts_with": True,
                },
                {
                    "letter": "T",
                    "hint": "Instrument used to measure temperature",
                    "answer": "Thermometer",
                    "starts_with": True,
                },
                {
                    "letter": "V",
                    "hint": "Planet often called Earth's twin",
                    "answer": "Venus",
                    "starts_with": True,
                },
                {
                    "letter": "Z",
                    "hint": "Striped equine native to Africa",
                    "answer": "Zebra",
                    "starts_with": True,
                },
            ],
        },
    ),
    # ── Science & Nature · ES ─────────────────────────────────────
    _example_doc(
        "word-pass",
        "es",
        "Science & Nature",
        {
            "game_type": "word-pass",
            "title": "Pasapalabra — Ciencia y Naturaleza",
            "difficulty_percentage": 50,
            "words": [
                {
                    "letter": "A",
                    "hint": "Unidad fundamental de la materia",
                    "answer": "Átomo",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Ciencia que estudia los seres vivos",
                    "answer": "Biología",
                    "starts_with": True,
                },
                {
                    "letter": "C",
                    "hint": "Unidad básica de los seres vivos",
                    "answer": "Célula",
                    "starts_with": True,
                },
                {
                    "letter": "D",
                    "hint": "Molécula que almacena la información genética",
                    "answer": "ADN",
                    "starts_with": False,
                },
                {
                    "letter": "E",
                    "hint": "Capacidad de un sistema para realizar trabajo",
                    "answer": "Energía",
                    "starts_with": True,
                },
                {
                    "letter": "F",
                    "hint": "Cambio de estado de sólido a líquido",
                    "answer": "Fusión",
                    "starts_with": True,
                },
                {
                    "letter": "G",
                    "hint": "Estudio de las rocas y la estructura terrestre",
                    "answer": "Geología",
                    "starts_with": True,
                },
                {
                    "letter": "H",
                    "hint": "Compuesto químico formado por dos átomos de hidrógeno y uno de oxígeno",
                    "answer": "H₂O",
                    "starts_with": True,
                },
                {
                    "letter": "I",
                    "hint": "Propiedad de un cuerpo de resistir cambios en su movimiento",
                    "answer": "Inercia",
                    "starts_with": True,
                },
                {
                    "letter": "J",
                    "hint": "Unidad de energía en el Sistema Internacional",
                    "answer": "Julio",
                    "starts_with": True,
                },
                {
                    "letter": "L",
                    "hint": "Radiación electromagnética visible al ojo humano",
                    "answer": "Luz",
                    "starts_with": True,
                },
                {
                    "letter": "M",
                    "hint": "Dispositivo que convierte energía eléctrica en movimiento",
                    "answer": "Motor",
                    "starts_with": True,
                },
                {
                    "letter": "N",
                    "hint": "Parte central del átomo que contiene protones y neutrones",
                    "answer": "Núcleo",
                    "starts_with": True,
                },
                {
                    "letter": "O",
                    "hint": "Capa de gas que protege la Tierra de la radiación ultravioleta",
                    "answer": "Ozono",
                    "starts_with": True,
                },
                {
                    "letter": "P",
                    "hint": "Tabla que clasifica los elementos químicos",
                    "answer": "Periódica",
                    "starts_with": True,
                },
                {
                    "letter": "R",
                    "hint": "Transferencia de energía sin medio material de propagación",
                    "answer": "Radiación",
                    "starts_with": True,
                },
                {
                    "letter": "S",
                    "hint": "Conjunto organizado de planetas orbitando una estrella",
                    "answer": "Sistema solar",
                    "starts_with": True,
                },
                {
                    "letter": "T",
                    "hint": "Magnitud que mide el nivel de calor de un cuerpo",
                    "answer": "Temperatura",
                    "starts_with": True,
                },
                {
                    "letter": "V",
                    "hint": "Rapidez con que un objeto cambia de posición",
                    "answer": "Velocidad",
                    "starts_with": True,
                },
                {
                    "letter": "Z",
                    "hint": "Ciencia que estudia los animales",
                    "answer": "Zoología",
                    "starts_with": True,
                },
            ],
        },
    ),
    # ── History · ES ──────────────────────────────────────────────
    _example_doc(
        "word-pass",
        "es",
        "History",
        {
            "game_type": "word-pass",
            "title": "Pasapalabra — Historia Universal",
            "difficulty_percentage": 55,
            "words": [
                {
                    "letter": "A",
                    "hint": "Civilización que construyó pirámides en Tenochtitlán",
                    "answer": "Azteca",
                    "starts_with": True,
                },
                {
                    "letter": "B",
                    "hint": "Muro que dividió una ciudad europea entre 1961 y 1989",
                    "answer": "Berlín",
                    "starts_with": True,
                },
                {
                    "letter": "C",
                    "hint": "Navegante genovés que llegó a América en 1492",
                    "answer": "Colón",
                    "starts_with": True,
                },
                {
                    "letter": "D",
                    "hint": "Periodo histórico también llamado Edad Oscura",
                    "answer": "Edad Media",
                    "starts_with": False,
                },
                {
                    "letter": "E",
                    "hint": "País donde nació la civilización faraónica",
                    "answer": "Egipto",
                    "starts_with": True,
                },
                {
                    "letter": "F",
                    "hint": "Revolución que comenzó en 1789",
                    "answer": "Francesa",
                    "starts_with": True,
                },
                {
                    "letter": "G",
                    "hint": "Imperio que duró del siglo XV al XX con capital en Estambul",
                    "answer": "Otomano",
                    "starts_with": False,
                },
                {
                    "letter": "H",
                    "hint": "Escritura jeroglífica fue creada por esta civilización del Nilo",
                    "answer": "Hieroglífico",
                    "starts_with": True,
                },
                {
                    "letter": "I",
                    "hint": "Revolución tecnológica que comenzó en Gran Bretaña en el siglo XVIII",
                    "answer": "Industrial",
                    "starts_with": True,
                },
                {
                    "letter": "J",
                    "hint": "Jefa de estado francesa ejecutada en la Revolución",
                    "answer": "Juana de Arco",
                    "starts_with": True,
                },
                {
                    "letter": "L",
                    "hint": "Filósofo griego autor de 'La República'",
                    "answer": "Platón",
                    "starts_with": False,
                },
                {
                    "letter": "M",
                    "hint": "Civilización precolombina de la península de Yucatán",
                    "answer": "Maya",
                    "starts_with": True,
                },
                {
                    "letter": "N",
                    "hint": "Líder militar francés que se coronó emperador en 1804",
                    "answer": "Napoleón",
                    "starts_with": True,
                },
                {
                    "letter": "O",
                    "hint": "Juegos de la Antigua Grecia celebrados cada cuatro años",
                    "answer": "Olímpicos",
                    "starts_with": True,
                },
                {
                    "letter": "P",
                    "hint": "Sistema de gobierno de la antigua Atenas basado en el voto ciudadano",
                    "answer": "Democracia",
                    "starts_with": False,
                },
                {
                    "letter": "R",
                    "hint": "Periodo cultural europeo que siguió a la Edad Media",
                    "answer": "Renacimiento",
                    "starts_with": True,
                },
                {
                    "letter": "S",
                    "hint": "Civilización mesopotámica considerada la más antigua",
                    "answer": "Sumeria",
                    "starts_with": True,
                },
                {
                    "letter": "T",
                    "hint": "Barco que se hundió en su viaje inaugural en 1912",
                    "answer": "Titanic",
                    "starts_with": True,
                },
                {
                    "letter": "V",
                    "hint": "Pueblo escandinavo famoso por sus expediciones marítimas",
                    "answer": "Vikingo",
                    "starts_with": True,
                },
                {
                    "letter": "Z",
                    "hint": "Antiguo nombre del río Congo en la época colonial",
                    "answer": "Zaire",
                    "starts_with": True,
                },
            ],
        },
    ),
]


# =====================================================================
#  TRUE/FALSE EXAMPLES
# =====================================================================

TRUE_FALSE_EXAMPLES: list[dict[str, Any]] = [
    _example_doc(
        "true_false",
        "es",
        "General Knowledge",
        {
            "game_type": "true_false",
            "title": "Verdadero o Falso — Cultura General",
            "difficulty_percentage": 40,
            "statements": [
                {
                    "statement": "El Sol gira alrededor de la Tierra.",
                    "is_true": False,
                    "explanation": "Es la Tierra la que orbita alrededor del Sol, modelo heliocéntrico demostrado por Copérnico.",
                },
                {
                    "statement": "El agua hierve a 100 °C al nivel del mar.",
                    "is_true": True,
                    "explanation": "A presión atmosférica estándar (1 atm), el punto de ebullición del agua es 100 °C.",
                },
                {
                    "statement": "Los murciélagos son ciegos.",
                    "is_true": False,
                    "explanation": "Los murciélagos pueden ver; además usan ecolocalización para orientarse en la oscuridad.",
                },
                {
                    "statement": "Japón está formado por un archipiélago.",
                    "is_true": True,
                    "explanation": "Japón consta de más de 6.800 islas, siendo las cuatro principales Honshū, Hokkaidō, Kyūshū y Shikoku.",
                },
                {
                    "statement": "El diamante es el material natural más duro.",
                    "is_true": True,
                    "explanation": "El diamante alcanza 10 en la escala de Mohs, la máxima dureza mineral conocida.",
                },
            ],
        },
    ),
    _example_doc(
        "true_false",
        "en",
        "Science & Nature",
        {
            "game_type": "true_false",
            "title": "True or False — Science",
            "difficulty_percentage": 45,
            "statements": [
                {
                    "statement": "Light travels faster than sound.",
                    "is_true": True,
                    "explanation": "Light travels at approximately 300,000 km/s while sound travels at about 343 m/s in air.",
                },
                {
                    "statement": "Humans use only 10% of their brain.",
                    "is_true": False,
                    "explanation": "Brain imaging shows that all regions of the brain have known functions and are active over time.",
                },
                {
                    "statement": "Water is composed of hydrogen and oxygen.",
                    "is_true": True,
                    "explanation": "Water (H₂O) consists of two hydrogen atoms bonded to one oxygen atom.",
                },
                {
                    "statement": "Electrons are larger than protons.",
                    "is_true": False,
                    "explanation": "Protons are about 1,836 times more massive than electrons.",
                },
                {
                    "statement": "The chemical symbol for iron is Fe.",
                    "is_true": True,
                    "explanation": "Fe comes from the Latin word 'ferrum' meaning iron.",
                },
            ],
        },
    ),
]


# =====================================================================
#  EDUCATIONAL RESOURCES (factual content by category)
# =====================================================================

EDUCATIONAL_RESOURCES: list[dict[str, Any]] = [
    # ── General Knowledge ─────────────────────────────────────────
    _resource_doc(
        "General Knowledge",
        "es",
        """
Cultura general abarca un amplio espectro de conocimientos sobre el mundo.
El planeta Tierra tiene una circunferencia de aproximadamente 40.075 km en el ecuador.
La población mundial superó los 8.000 millones de personas en 2022.
El idioma más hablado del mundo por número total de hablantes es el inglés,
seguido del chino mandarín y el hindi.
Las Naciones Unidas tienen 193 estados miembros.
El sistema métrico decimal fue adoptado oficialmente en Francia en 1795.
Internet se originó del proyecto ARPANET en 1969.
La Estación Espacial Internacional orbita la Tierra a unos 408 km de altitud.
""",
    ),
    _resource_doc(
        "General Knowledge",
        "en",
        """
General knowledge spans a broad spectrum of facts about the world.
Earth has a circumference of roughly 40,075 km at the equator.
The world population surpassed 8 billion people in 2022.
The most spoken language worldwide by total number of speakers is English,
followed by Mandarin Chinese and Hindi.
The United Nations has 193 member states.
The metric system was officially adopted in France in 1795.
The Internet originated from the ARPANET project in 1969.
The International Space Station orbits Earth at about 408 km altitude.
""",
    ),
    # ── Science & Nature ──────────────────────────────────────────
    _resource_doc(
        "Science & Nature",
        "es",
        """
La tabla periódica tiene 118 elementos confirmados. El hidrógeno (H) es el más ligero
y abundante del universo. El oro (Au) tiene número atómico 79.
La velocidad de la luz en el vacío es 299.792.458 metros por segundo.
La fotosíntesis convierte CO₂ y agua en glucosa y oxígeno usando luz solar.
El ADN (ácido desoxirribonucleico) almacena la información genética de los seres vivos.
Los humanos comparten aproximadamente el 98,7% de su ADN con los chimpancés.
La célula es la unidad básica de la vida; existen células procariotas y eucariotas.
La gravedad en la superficie de la Luna es aproximadamente 1/6 de la terrestre.
Newton formuló las tres leyes del movimiento en 1687.
""",
    ),
    _resource_doc(
        "Science & Nature",
        "en",
        """
The periodic table contains 118 confirmed elements. Hydrogen (H) is the lightest
and most abundant element in the universe. Gold (Au) has atomic number 79.
The speed of light in a vacuum is 299,792,458 metres per second.
Photosynthesis converts CO₂ and water into glucose and oxygen using sunlight.
DNA (deoxyribonucleic acid) stores the genetic information of living organisms.
Humans share approximately 98.7% of their DNA with chimpanzees.
The cell is the basic unit of life; cells are either prokaryotic or eukaryotic.
Surface gravity on the Moon is roughly 1/6 of Earth's.
Newton formulated his three laws of motion in 1687.
""",
    ),
    # ── History ───────────────────────────────────────────────────
    _resource_doc(
        "History",
        "es",
        """
La Revolución Francesa comenzó en 1789 con la toma de la Bastilla el 14 de julio.
El Imperio Romano de Occidente cayó en el 476 d.C. con la deposición de Rómulo Augústulo.
Cristóbal Colón llegó a América el 12 de octubre de 1492 financiado por los Reyes Católicos.
La Primera Guerra Mundial (1914-1918) fue causada en parte por el asesinato del archiduque
Francisco Fernando de Austria en Sarajevo.
La Segunda Guerra Mundial terminó en 1945 con la rendición de Japón.
El Muro de Berlín cayó el 9 de noviembre de 1989.
La imprenta de tipos móviles fue inventada por Johannes Gutenberg alrededor de 1440.
La civilización sumeria, en Mesopotamia, es considerada una de las más antiguas (c. 4500 a.C.).
""",
    ),
    _resource_doc(
        "History",
        "en",
        """
The French Revolution began in 1789 with the storming of the Bastille on 14 July.
The Western Roman Empire fell in 476 AD with the deposition of Romulus Augustulus.
Christopher Columbus reached the Americas on 12 October 1492, funded by the Spanish Crown.
World War I (1914-1918) was partly triggered by the assassination of Archduke
Franz Ferdinand of Austria in Sarajevo.
World War II ended in 1945 with the surrender of Japan.
The Berlin Wall fell on 9 November 1989.
The movable-type printing press was invented by Johannes Gutenberg around 1440.
The Sumerian civilisation in Mesopotamia is among the oldest known (c. 4500 BC).
""",
    ),
    # ── Sports ────────────────────────────────────────────────────
    _resource_doc(
        "Sports",
        "es",
        """
El fútbol es el deporte más popular del mundo con más de 4.000 millones de aficionados.
La Copa Mundial de la FIFA se celebra cada cuatro años desde 1930.
Los Juegos Olímpicos modernos comenzaron en Atenas en 1896 por iniciativa de Pierre de Coubertin.
Una maratón mide exactamente 42,195 km, distancia estandarizada en 1921.
El baloncesto fue inventado por James Naismith en 1891 en Springfield, Massachusetts.
El cricket es especialmente popular en India, Australia e Inglaterra.
El Tour de Francia es la carrera ciclista más prestigiosa, creada en 1903.
El tenis tiene cuatro torneos de Grand Slam: Australian Open, Roland Garros, Wimbledon y US Open.
""",
    ),
    # ── Geography ─────────────────────────────────────────────────
    _resource_doc(
        "Geography",
        "es",
        """
La Tierra tiene siete continentes: África, Antártida, Asia, Europa, América del Norte,
América del Sur y Oceanía.
El Monte Everest mide 8.849 metros sobre el nivel del mar (medición 2020).
El río Nilo, con aproximadamente 6.650 km, es considerado el más largo del mundo.
Rusia es el país más grande por superficie (17,1 millones de km²).
El desierto del Sahara es el desierto cálido más grande, con ~9 millones de km².
La Fosa de las Marianas es el punto más profundo del océano, a ~11.034 metros.
El lago Baikal en Rusia es el lago más profundo del mundo (1.642 m).
Canberra es la capital de Australia, no Sídney ni Melbourne.
""",
    ),
    # ── Art ───────────────────────────────────────────────────────
    _resource_doc(
        "Art",
        "es",
        """
Leonardo da Vinci pintó la Mona Lisa (La Gioconda) entre 1503 y 1519;
se exhibe en el Museo del Louvre de París.
Vincent van Gogh pintó La noche estrellada en 1889 durante su internamiento en Saint-Rémy.
Miguel Ángel esculpió el David de mármol entre 1501 y 1504; mide 5,17 metros.
El impresionismo nació en París a partir de la exposición de 1874 con obras de Monet, Renoir y Degas.
Pablo Picasso y Georges Braque desarrollaron el cubismo a principios del siglo XX.
Salvador Dalí fue uno de los máximos exponentes del surrealismo.
La arquitectura gótica se caracteriza por arcos ojivales, bóvedas de crucería y vidrieras.
El Museo del Prado en Madrid alberga obras de Velázquez, Goya y El Greco.
""",
    ),
    # ── Animals ───────────────────────────────────────────────────
    _resource_doc(
        "Animals",
        "es",
        """
La ballena azul es el animal más grande que ha existido, con hasta 30 metros de longitud.
El guepardo es el animal terrestre más rápido, alcanzando 112 km/h.
Las arañas son arácnidos con 8 patas; los insectos tienen 6.
Los delfines son mamíferos marinos que respiran aire y amamantan a sus crías.
Los camaleones cambian de color mediante células cromatóforas en su piel.
Los koalas se alimentan casi exclusivamente de hojas de eucalipto.
Las abejas son polinizadores esenciales; una colmena puede tener 60.000 individuos.
El pulpo tiene tres corazones y sangre azul (basada en cobre, no hierro).
""",
    ),
    # ── Entertainment: Film ───────────────────────────────────────
    _resource_doc(
        "Entertainment: Film",
        "en",
        """
The Academy Awards (Oscars) have been presented annually since 1929.
Toy Story (1995) by Pixar was the first fully computer-animated feature film.
The Marvel Cinematic Universe (MCU) began with Iron Man in 2008.
Alfred Hitchcock is known as the "Master of Suspense" for films like Psycho and Vertigo.
Quentin Tarantino directed Pulp Fiction (1994), which won the Palme d'Or at Cannes.
Studio Ghibli, founded by Hayao Miyazaki, produced classics like Spirited Away (2001).
The highest-grossing film of all time (adjusted for inflation) remains Gone with the Wind (1939).
""",
    ),
    # ── Computers ─────────────────────────────────────────────────
    _resource_doc(
        "Science: Computers",
        "es",
        """
La primera computadora electrónica de propósito general fue ENIAC (1945).
El lenguaje de programación Python fue creado por Guido van Rossum en 1991.
La World Wide Web fue inventada por Tim Berners-Lee en 1989 en el CERN.
Un byte equivale a 8 bits; un kilobyte son 1.024 bytes.
Linux es un sistema operativo de código abierto creado por Linus Torvalds en 1991.
La inteligencia artificial busca crear sistemas capaces de realizar tareas que requieren
inteligencia humana, como reconocimiento de voz y toma de decisiones.
JavaScript es el lenguaje de programación más utilizado en desarrollo web.
""",
    ),
    # ── Mathematics ───────────────────────────────────────────────
    _resource_doc(
        "Science: Mathematics",
        "es",
        """
El número pi (π) es la relación entre la circunferencia y el diámetro de un círculo,
aproximadamente 3,14159.
El teorema de Pitágoras establece que en un triángulo rectángulo a² + b² = c².
Los números primos solo son divisibles por 1 y por sí mismos.
El cero fue usado como número por primera vez por los matemáticos indios en el siglo V.
La sucesión de Fibonacci es: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34...
El álgebra proviene del árabe 'al-jabr', del libro de Al-Juarismi del siglo IX.
Euler demostró que e^(iπ) + 1 = 0, considerada la fórmula más bella de las matemáticas.
""",
    ),
    # ── Mythology ─────────────────────────────────────────────────
    _resource_doc(
        "Mythology",
        "es",
        """
En la mitología griega, Zeus era el rey de los dioses olímpicos.
Poseidón era el dios del mar y Hades gobernaba el inframundo.
Los doce trabajos de Hércules son hazañas legendarias impuestas como penitencia.
La mitología nórdica incluye dioses como Odín, Thor y Loki.
El Ragnarök es el fin del mundo en la mitología nórdica.
En la mitología egipcia, Ra era el dios del sol y Anubis el dios de los muertos.
La mitología romana adoptó muchos dioses griegos: Zeus = Júpiter, Ares = Marte.
El Minotauro vivía en el laberinto de Creta, según la leyenda griega.
""",
    ),
    # ── Vehicles ──────────────────────────────────────────────────
    _resource_doc(
        "Vehicles",
        "es",
        """
Karl Benz patentó el primer automóvil con motor de combustión interna en 1886.
El Ford Model T (1908) fue el primer coche producido en masa con línea de ensamblaje.
Los coches eléctricos usan baterías de iones de litio; Tesla popularizó el concepto desde 2008.
El Concorde fue un avión supersónico comercial que volaba a Mach 2,04.
El Shinkansen japonés (tren bala) comenzó a operar en 1964 a velocidades de 210 km/h.
Los motores diésel fueron inventados por Rudolf Diesel en 1893.
El Airbus A380 es el avión de pasajeros más grande del mundo con capacidad para 853 personas.
""",
    ),
    # ── Entertainment: Music ──────────────────────────────────────
    _resource_doc(
        "Entertainment: Music",
        "es",
        """
Ludwig van Beethoven compuso nueve sinfonías; la Novena incluye el «Himno a la Alegría».
The Beatles, formados en Liverpool en 1960, son la banda más vendedora de la historia.
Wolfgang Amadeus Mozart compuso más de 600 obras antes de morir a los 35 años.
El jazz nació en Nueva Orleans a principios del siglo XX fusionando blues, ragtime y gospel.
La guitarra eléctrica fue popularizada en los años 1950 por músicos como Chuck Berry.
El reggae se originó en Jamaica en los años 1960; Bob Marley es su figura más icónica.
La ópera nació en Italia a finales del siglo XVI; La Camerata Fiorentina fue pionera.
""",
    ),
    # ── Politics ──────────────────────────────────────────────────
    _resource_doc(
        "Politics",
        "es",
        """
La democracia se originó en la Antigua Atenas en el siglo V a.C.
La Declaración Universal de los Derechos Humanos fue adoptada por la ONU en 1948.
La Unión Europea fue fundada formalmente con el Tratado de Maastricht en 1992.
El sistema de gobierno de Estados Unidos se basa en la separación de poderes:
ejecutivo, legislativo y judicial.
La ONU tiene cinco miembros permanentes del Consejo de Seguridad: EE.UU., Rusia,
China, Francia y Reino Unido.
El sufragio universal se fue implementando progresivamente durante los siglos XIX y XX.
""",
    ),
]


# =====================================================================
#  COMPLETE CORPUS — single import point
# =====================================================================


def get_full_corpus() -> list[dict[str, Any]]:
    """Return every document in the corpus (examples + resources)."""
    return (
        QUIZ_EXAMPLES + WORD_PASS_EXAMPLES + TRUE_FALSE_EXAMPLES + EDUCATIONAL_RESOURCES
    )


def get_corpus_signature() -> str:
    """Return a stable content signature for the curated corpus."""
    normalized = [
        {
            "doc_id": entry.get("doc_id"),
            "content": entry.get("content"),
            "metadata": entry.get("metadata", {}),
        }
        for entry in get_full_corpus()
    ]
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
