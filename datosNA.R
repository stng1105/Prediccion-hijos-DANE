#selección de variables que no tengan muchas datos NA

seleccionar<- function(datos){
   output <- vector()  
   for (i in seq_along(datos)) {          
      a= (mean(is.na(datos[[i]])))
      if(a<0.4){
         output <- c(output, colnames(datos[i]))
      }
      
      
   }
   output
}


