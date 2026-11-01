#pragma once

// Deterministic circular gravitational billiards. Position and velocity are
// double precision in metres and seconds; no RNG, damping, or particle-particle
// interaction. Ball trajectories are independent and may overlap.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace reaction::billiards {

struct Ball {
    double x=0.0, y=0.0, vx=0.0, vy=0.0;
    std::uint64_t impacts=0;
};

inline double energy(const Ball& b,double gravity) noexcept {
    return 0.5*(b.vx*b.vx+b.vy*b.vy)+gravity*b.y;
}

inline Ball ballistic(const Ball& b,double t,double gravity) noexcept {
    return {b.x+b.vx*t,b.y+b.vy*t-0.5*gravity*t*t,
            b.vx,b.vy-gravity*t,b.impacts};
}

inline double wallDistanceSquared(const Ball& b) noexcept {
    return b.x*b.x+b.y*b.y;
}

// Advance one small time interval analytically under uniform acceleration.
// When it crosses the circular wall, bisect the first crossing, reflect about
// the exact circle normal and propagate the remaining fraction of the interval.
// A small fixed outer timestep ensures crossings are never jumped over during
// ordinary interior trajectories. No noise or artificial velocity kicks.
inline void advance(Ball& b,double dt,double gravity,double usableRadius) noexcept {
    if(!(dt>0.0) || !std::isfinite(dt) || !(gravity>0.0) ||
       !(usableRadius>0.0)) return;
    const double radius2=usableRadius*usableRadius;
    double remaining=dt;
    for(int collision=0;collision<8 && remaining>0.0;collision++) {
        const Ball end=ballistic(b,remaining,gravity);
        if(wallDistanceSquared(end)<=radius2) {
            b=end;
            return;
        }
        double low=0.0,high=remaining;
        for(int j=0;j<55;j++) {
            const double mid=(low+high)*0.5;
            if(wallDistanceSquared(ballistic(b,mid,gravity))>radius2) high=mid;
            else low=mid;
        }
        const double hitTime=(low+high)*0.5;
        Ball at=ballistic(b,hitTime,gravity);
        const double length=std::hypot(at.x,at.y);
        if(length<1e-15) { b=at; return; }
        const double nx=at.x/length,ny=at.y/length;
        at.x=nx*usableRadius;
        at.y=ny*usableRadius;
        const double normalVelocity=at.vx*nx+at.vy*ny;
        at.vx-=2.0*normalVelocity*nx;
        at.vy-=2.0*normalVelocity*ny;
        // The launch energy is 0 (every particle is released from rest
        // at y=0). Renormalize only roundoff at impact, preserving it to
        // machine precision across thousands of reflections.
        const double speed=std::hypot(at.vx,at.vy);
        const double idealSpeed=std::sqrt(std::max(0.0,-2.0*gravity*at.y));
        if(speed>1e-14){
            at.vx*=idealSpeed/speed;
            at.vy*=idealSpeed/speed;
        }
        ++at.impacts;
        b=at;
        remaining-=hitTime;
        // Move a few ulps inside the wall to make f(0)<0 next iteration.
        b.x*=1.0-2.0e-15;
        b.y*=1.0-2.0e-15;
    }
}

// Exact sample positions: width is the total fan span, not nearest-neighbor
// displacement. 200 balls over 0.1mm => 0.5025 micrometre separation.
inline Ball initialBall(int index,int count,double fanWidthMeters) noexcept {
    const double u=count>1 ? double(index)/double(count-1)-0.5 : 0.0;
    return {u*fanWidthMeters,0.0,0.0,0.0,0};
}

} // namespace reaction::billiards
