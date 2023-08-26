#include <optixu/optixu_math_namespace.h>
#include "redflash.h"
#include "random.h"
#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float, time, , );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, aabb_min, , );
rtDeclareVariable(float3, aabb_max, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

// プライマリレイのDepthを利用した高速化用
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

static __forceinline__ __device__ float3 abs_float3(float3 v)
{
    return make_float3(abs(v.x), abs(v.y), abs(v.z));
}

static __forceinline__ __device__ float3 max_float3(float3 v, float a)
{
    return make_float3(max(v.x, a), max(v.y, a), max(v.z, a));
}

static __forceinline__ __device__ float dMenger(float3 z0, float3 offset, float scale) {
    float3 z = z0;
    float w = 1.0;
    float scale_minus_one = scale - 1.0;

    for (int n = 0; n < 3; n++) {
        z = abs_float3(z);

        // if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.y)
        {
            float x = z.x;
            z.x = z.y;
            z.y = x;
        }

        // if (z.x < z.z) z.xz = z.zx;
        if (z.x < z.z)
        {
            float x = z.x;
            z.x = z.z;
            z.z = x;
        }

        // if (z.y < z.z) z.yz = z.zy;
        if (z.y < z.z)
        {
            float y = z.y;
            z.y = z.z;
            z.z = y;
        }

        z *= scale;
        w *= scale;

        z -= offset * scale_minus_one;

        float tmp = offset.z * scale_minus_one;
        if (z.z < -0.5 * tmp) z.z += tmp;
    }
    return (length(max_float3(abs_float3(z) - make_float3(1.0), 0.0)) - 0.01) / w;
}

float3 get_xyz(float4 p)
{
    return make_float3(p.x, p.y, p.z);
}

// not work...
void set_xyz(float4& a, float3 b)
{
    a.x = b.x;
    a.y = b.y;
    a.x = b.z;
}

float dMandelFast(float3 p, float scale, int n) {
    float4 q0 = make_float4(p, 1.);
    float4 q = q0;

    for (int i = 0; i < n; i++) {
        // q.xyz = clamp(q.xyz, -1.0, 1.0) * 2.0 - q.xyz;
        // set_xyz(q, clamp(get_xyz(q), -1.0, 1.0) * 2.0 - get_xyz(q));
        float4 tmp = clamp(q, -1.0, 1.0) * 2.0 - q;
        q.x = tmp.x;
        q.y = tmp.y;
        q.z = tmp.z;

        // q = q * scale / clamp( dot( q.xyz, q.xyz ), 0.3, 1.0 ) + q0;
        float3 q_xyz = get_xyz(q);
        q = q * scale / clamp(dot(q_xyz, q_xyz), 0.3, 1.0) + q0;
    }

    // return length( q.xyz ) / abs( q.w );
    return length(get_xyz(q)) / abs(q.w);
}

float fracf(float x)
{
    return x - floor(x);
}

float mod(float a, float b)
{
    return fracf(abs(a / b)) * abs(b);
}

float opRep(float p, float interval)
{
    return mod(p, interval) - interval * 0.5;
}

float map(float3 p)
{
    // return dMenger((p - center) / scale, make_float3(1.23, 1.65, 1.45), 2.56) * scale;
    // return dMenger((p - center) / scale, make_float3(1, 1, 1), 3.1) * scale;

    if (time < 7)
    {
        float s = 2.76;
        float t;

        if (time < 3.5)
        {
            // nop
        }
        else if (time < 5)
        {
            t = time - 3.5;
            s = 3.1 - 0.2 * t;
        }
        else
        {
            t = time - 5;
            s = 2.8 - 0.2 * t;
        }

        float scale = 70.0f;
        return dMandelFast((p - center) / scale, s, 20) * scale;
    }
    else
    {
        float scale = 100;
        return dMenger((p - center) / scale, make_float3(1.2, 1.0, 1.2 + 0.6 * sin(time + 6.2)), 2.8 + 0.1 * sin(time + 3.2)) * scale;
    }
}

#define calcNormal(p, dFunc, eps) normalize(\
    make_float3( eps, -eps, -eps) * dFunc(p + make_float3( eps, -eps, -eps)) + \
    make_float3(-eps, -eps,  eps) * dFunc(p + make_float3(-eps, -eps,  eps)) + \
    make_float3(-eps,  eps, -eps) * dFunc(p + make_float3(-eps,  eps, -eps)) + \
    make_float3( eps,  eps,  eps) * dFunc(p + make_float3( eps,  eps,  eps)))

float3 calcNormalBasic(float3 p, float eps)
{
    return normalize(make_float3(
        map(p + make_float3(eps, 0.0, 0.0)) - map(p + make_float3(-eps, 0.0, 0.0)),
        map(p + make_float3(0.0, eps, 0.0)) - map(p + make_float3(0.0, -eps, 0.0)),
        map(p + make_float3(0.0, 0.0, eps)) - map(p + make_float3(0.0, 0.0, -eps))
    ));
}

// https://www.shadertoy.com/view/lttGDn
float calcEdge(float3 p, float width)
{
    float edge = 0.0;
    float2 e = make_float2(width, 0.0f);

    // Take some distance function measurements from either side of the hit point on all three axes.
    float d1 = map(p + make_float3(width, 0.0f, 0.0f)), d2 = map(p - make_float3(width, 0.0f, 0.0f));
    float d3 = map(p + make_float3(0.0f, width, 0.0f)), d4 = map(p - make_float3(0.0f, width, 0.0f));
    float d5 = map(p + make_float3(0.0f, 0.0f, width)), d6 = map(p - make_float3(0.0f, 0.0f, width));
    float d = map(p) * 2.;	// The hit point itself - Doubled to cut down on calculations. See below.

    // Edges - Take a geometry measurement from either side of the hit point. Average them, then see how
    // much the value differs from the hit point itself. Do this for X, Y and Z directions. Here, the sum
    // is used for the overall difference, but there are other ways. Note that it's mainly sharp surface
    // curves that register a discernible difference.
    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = max(max(abs(d1 + d2 - d), abs(d3 + d4 - d)), abs(d5 + d6 - d)); // Etc.

    // Once you have an edge value, it needs to normalized, and smoothed if possible. How you
    // do that is up to you. This is what I came up with for now, but I might tweak it later.
    edge = smoothstep(0., 1., sqrt(edge / e.x * 2.));

    // Return the normal.
    // Standard, normalized gradient mearsurement.
    return edge;
}

RT_CALLABLE_PROGRAM void materialAnimation_Nop(MaterialParameter& mat, State& state)
{
    // nop
}

RT_CALLABLE_PROGRAM void materialAnimation_Raymarching(MaterialParameter& mat, State& state)
{
    if (time < 7) return;

    // MengerSpongeのシーンのEmissiveアニメーション
    float3 p = state.hitpoint;
    float edge = calcEdge(p, 0.02);
    mat.emission = make_float3(0.2, 0.2, 1) * pow(edge, 2.0f) * abs(sin(0.1 * p.z + 0.5 * time));

    float bar = smoothstep(0.7, 1.0, sin(p.z + 2 * time));
    if (state.normal.y > 0.8) bar = 0;
    mat.emission += bar * make_float3(0.2, 0.2, 20);

    mat.roughness = 0.005;
    mat.metallic = 0.5;
    mat.albedo = make_float3(0.8);
}

RT_PROGRAM void intersect(int primIdx)
{
    float eps;
    float t = ray.tmin, d = 0.0;
    float3 p = ray.origin;

    if (current_prd.depth == 0)
    {
        t = max(current_prd.distance, t);
    }

    for (int i = 0; i < 300; i++)
    {
        p = ray.origin + t * ray.direction;
        d = map(p);
        t += d;
        eps = scene_epsilon * t;
        if (abs(d) < eps || t > ray.tmax)
        {
            break;
        }
    }

    if (t < ray.tmax && rtPotentialIntersection(t))
    {
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon);
        texcoord = make_float3(p.x, p.y, 0);
        rtReportIntersection(0);
    }
}

float calcSlope(float t0, float t1, float r0, float r1)
{
    return (r1 - r0) / max(t1 - t0, 1e-5);
}

RT_PROGRAM void intersect_AutoRelaxation(int primIdx)
{
    float t = ray.tmin;

    if (current_prd.depth == 0)
    {
        t = max(current_prd.distance, t);
    }

    float eps = scene_epsilon * t;
    float r = map(ray.origin + t * ray.direction);
    int i = 1;
    float z = r;
    float m = -1;
    float stepRelaxation = 0.2;

    while (t + r < ray.tmax          // miss
        && r > eps    // hit
        && i < 300)  // didn't converge
    {
        float T = t + z;
        float R = map(ray.origin + T * ray.direction);
        bool doBackStep = z > abs(R) + r;
        //bool doBackStep = t + abs(r) < T - abs(R);
        float M = calcSlope(t, T, r, R);
        m = doBackStep ? -1 : lerp(m, M, stepRelaxation);
        t = doBackStep ? t : T;
        r = doBackStep ? r : R;
        float omega = max(1.0, 2.0 / (1.0 - m));
        eps = scene_epsilon * t;
        z = max(eps, r * omega);
        ++i;
#ifdef ENABLE_DEBUG_UTILS
        // backStep += doBackStep ? 1 : 0;
#endif
    }

#ifdef ENABLE_DEBUG_UTILS
    // stepCount = i;
#endif

    float retT = t + r;
    //retT = min(retT, ray.tmax);

    if (retT < ray.tmax && rtPotentialIntersection(retT))
    {
        float3 p = ray.origin + retT * ray.direction;
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon);
        texcoord = make_float3(p.x, p.y, 0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = aabb_min;
    aabb->m_max = aabb_max;
}