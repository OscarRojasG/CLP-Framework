def distribuir_suma_exacta(arreglo, suma_objetivo):
    total_original = sum(arreglo)
    if total_original == 0:
        return [0] * len(arreglo)

    # 1. Calcular cuotas exactas y separar parte entera y decimal
    cuotas_exactas = [(v * suma_objetivo) / total_original for v in arreglo]
    resultado = [int(q) for q in cuotas_exactas]
    
    # 2. ¿Cuánto nos falta para llegar a la suma_objetivo?
    diferencia = suma_objetivo - sum(resultado)
    
    # 3. Identificar los índices con los decimales más grandes
    # Guardamos (valor_decimal, índice)
    decimales = []
    for i, q in enumerate(cuotas_exactas):
        decimales.append((q - int(q), i))
    
    # Ordenamos por decimal de mayor a menor
    decimales.sort(reverse=True, key=lambda x: x[0])
    
    # 4. Repartir la diferencia sumando 1 a los que más decimal tenían
    for i in range(int(diferencia)):
        indice_a_incrementar = decimales[i][1]
        resultado[indice_a_incrementar] += 1
        
    return resultado