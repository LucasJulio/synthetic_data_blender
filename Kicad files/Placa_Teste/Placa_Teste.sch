EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L pspice:CAP C1
U 1 1 5E933509
P 4100 2900
F 0 "C1" H 4278 2946 50  0000 L CNN
F 1 "CAP" H 4278 2855 50  0000 L CNN
F 2 "Capacitor_THT:C_Radial_D5.0mm_H11.0mm_P2.00mm" H 4100 2900 50  0001 C CNN
F 3 "~" H 4100 2900 50  0001 C CNN
	1    4100 2900
	1    0    0    -1  
$EndComp
$Comp
L pspice:CAP C3
U 1 1 5E934259
P 5200 2750
F 0 "C3" H 5378 2796 50  0000 L CNN
F 1 "CAP" H 5378 2705 50  0000 L CNN
F 2 "Capacitor_THT:C_Radial_D4.0mm_H5.0mm_P1.50mm" H 5200 2750 50  0001 C CNN
F 3 "~" H 5200 2750 50  0001 C CNN
	1    5200 2750
	1    0    0    -1  
$EndComp
$Comp
L Diode:1N4148 D3
U 1 1 5E9362D2
P 5350 2000
F 0 "D3" H 5350 2216 50  0000 C CNN
F 1 "1N4148" H 5350 2125 50  0000 C CNN
F 2 "Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal" H 5350 1825 50  0001 C CNN
F 3 "https://assets.nexperia.com/documents/data-sheet/1N4148_1N4448.pdf" H 5350 2000 50  0001 C CNN
	1    5350 2000
	1    0    0    -1  
$EndComp
$Comp
L pspice:DIODE D7
U 1 1 5E937405
P 4350 2200
F 0 "D7" H 4350 2465 50  0000 C CNN
F 1 "DIODE" H 4350 2374 50  0000 C CNN
F 2 "Diode_THT:D_5W_P5.08mm_Vertical_AnodeUp" H 4350 2200 50  0001 C CNN
F 3 "~" H 4350 2200 50  0001 C CNN
	1    4350 2200
	1    0    0    -1  
$EndComp
Wire Wire Line
	4100 2650 4100 2200
Wire Wire Line
	4100 2200 4150 2200
Wire Wire Line
	4550 2200 5200 2200
Wire Wire Line
	5200 2200 5200 2000
Wire Wire Line
	5500 2000 5500 2500
Wire Wire Line
	5500 2500 5200 2500
Wire Wire Line
	4100 3150 5200 3150
Wire Wire Line
	5200 3150 5200 3000
$EndSCHEMATC
