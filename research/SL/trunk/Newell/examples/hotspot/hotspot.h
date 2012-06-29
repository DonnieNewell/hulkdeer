#ifndef HOTSPOT_H
#define HOTSPOT_H
void runHotspotSetData(float *, int);
void runHotspot(float *, int, int, int , float step_div_Cap, float Rx, float Ry,
                float Rz, int device);
void runHotspotCleanup();
#endif