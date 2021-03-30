library(shiny)

#======================================================================================#
# Definir el UI para la
# Inicio UI
ui <- fluidPage(
    
    # titulos
    titlePanel(
        "Predicción de Hijos en Hogares Colombianos basado en datos del DANE"
    ),
    h4("Introduce las variables correspondientes:"),
    
    # En side bar panel van todos los selectores
    sidebarLayout(
        sidebarPanel(
            #======================================================================================#
            #inicio selectores
            selectInput(
                "genero",
                "Genero del jefe del hogar:",
                c("Masculino" = 1,
                  "Femenino" = 2)
            ),
            sliderInput(
                "edad",
                "Edad del genero del hogar:",
                min = 1,
                max = 100,
                value = 30
            ),
            sliderInput(
                "personas",
                "Cantidad de personas que conforman el hogar:",
                min = 1,
                max = 15,
                value = 3
            ),
            numericInput(
                "ingresos",
                "Ingresos del hogar:",
                min = 0,
                max = 100000000,
                value = 850000
            ),
            selectInput(
                "estadocivil",
                "Estado civil del jefe del hogar:",
                c(
                    "Casado(a)" = 6,
                    "Soltero(a)" = 5,
                    "Está separado(a) o divorciado(a)" = 4,
                    "Viudo(a)" = 3,
                    "No está casado(a) y vive en pareja hace dos años o más" = 2,
                    "No está casado(a) y vive en pareja hace menos de dos años" = 1
                )
            ),
            sliderInput(
                "cuartos",
                "¿En cuántos cuartos duermen las personas de este hogar?",
                min = 1,
                max = 10,
                value = 2
            ),
            selectInput(
                "etnia",
                "¿A cuál pueblo o etnia indígena pertenece el jefe del hogar?",
                c(
                    "Ninguna" = 6,
                    "Indígena" = 1,
                    "Gitano (a) (Rom)" = 2,
                    "Raizal del archipiélago de San Andrés, Providencia y Santa Catalina" = 3,
                    "Palenquero (a) de San Basilio" = 4,
                    "Negro (a), mulato (a) (afrodescendiente), afrocolombiano(a)" = 5
                )
            )
            
            
            #fin selectores
            #======================================================================================#
            
        ),
        
        #======================================================================================#
        # Mostrar output de server
        mainPanel(htmlOutput("textoSeleccion"),
                  htmlOutput("textoPrediccion"))
        # Fin mostrar output
        #======================================================================================#
    )
)
# fin UI

#======================================================================================#
# Define server logic 

#inicio server
server <- function(input, output) {
    output$textoSeleccion <- renderText({
        paste("<p> Edad: ",
              input$edad,
              "<br></p>")
    })
    
    output$textoPrediccion <- renderText({
        paste("<p>Tu edad + 10: ",
              modelo(input$edad),
              "</p>")
    })
}
#fin server

#======================================================================================#
# funciones
modelo <- function(edad){
    #aca se puede llamar a un modelo
    return(edad+10)
}
#fin funciones
#======================================================================================#

# Run the application
shinyApp(ui = ui, server = server)
#======================================================================================#