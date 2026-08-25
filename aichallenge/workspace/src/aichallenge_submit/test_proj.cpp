#include <iostream>
#include <iomanip>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_io/Io.h>

int main() {
    lanelet::projection::UtmProjector projector(lanelet::Origin({35.625, 139.781}));
    lanelet::GPSPoint gps;
    gps.lat = 35.62581953015;
    gps.lon = 139.78142932763;
    gps.ele = 6.5;
    lanelet::BasicPoint3d p = projector.forward(gps);
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "p.x = " << p.x() << std::endl;
    std::cout << "p.y = " << p.y() << std::endl;
    return 0;
}
