#ifndef HOTSPOT_H
#define HOTSPOT_H
void runHotspotSetData(DTYPE *, int);
void runHotspot(DTYPE *, int, int, int, const int, float step_div_Cap, float Rx, float Ry,
                float Rz, int device);
void runHotspotCleanup();
#endif