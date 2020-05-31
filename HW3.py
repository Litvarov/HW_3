import pylab
import numpy.fft as fft
import numpy
import tools


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Шаг по пространству
    dx = 0.2e-3

    # Размер области моделирования
    x = 0.1

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Шаг по времени
    dt = Sc * dx / c
    print(f"Шаг временной сетки: {dt} с")

    # Время расчета в отсчетах
    maxTime = 700

    # Размер области моделирования в отсчетах
    maxSize = int(x / dx)

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize)

    for probe in probes:
        probe.addData(Ez, Hy)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    # Параметры гауссова импульса
    A0 = 100  # ослабление в 0 момент времени по отношение к максимуму
    Am = 100  # ослабление на частоте Fm
    Fm = 12e9
    wg = numpy.sqrt(numpy.log(Am)) / (numpy.pi * Fm)
    NWg = wg / dt
    dg = wg * numpy.sqrt(numpy.log(A0))
    NDg = dg / dt

    for t in range(1, maxTime):
        # Граничные условия для поля H
        Hy[-1] = Hy[-2]

        # Расчет компоненты поля H
        Ez_shift = Ez[1:]
        Hy[:-1] = Hy[:-1] + (Ez_shift - Ez[:-1]) * Sc / W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / W0) * \
            numpy.exp(-((t - NDg - sourcePos) / NWg) ** 2)
        # Hy[sourcePos - 1] -= (Sc / W0) * numpy.exp(-((t - NDg)/NWg) ** 2 )

        # Граничные условия для поля E
        Ez[0] = Ez[1]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:] = Ez[1:] + (Hy[1:] - Hy_shift) * Sc * W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += Sc * \
            numpy.exp(-(((t + 0.5) - (sourcePos - 0.5) - NDg) / NWg) ** 2)
        # Ez[sourcePos] += Sc * numpy.exp(-(((t + 0.5) - (-0.5) - NDg)/NWg) ** 2 )

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 50 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Взятие преобразование Фурье
    F1 = fft.fft(probe.E,2048)
    F1 = fft.fftshift(F1)
    # Определние шага по частоте
    df = 1 / (len(F1) * dt)
    freq = numpy.arange(-len(F1) / 2 *
           df, len(F1) / 2 * df, df)
    # Построение графика
    fig, ax = pylab.subplots()
    # Настройка внешнего вида графиков
    ax.set_xlim(0, 15e9)
    ax.set_xlabel('Частота, Гц')
    ax.set_ylabel('|S| / |Smax|')
    ax.grid()
    ax.plot(freq, abs(F1 / numpy.max(F1)))
    pylab.show()
