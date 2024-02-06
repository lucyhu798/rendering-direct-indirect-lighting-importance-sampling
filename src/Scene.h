#pragma once

#include <string>
#include <vector>
#include <limits>

#include <glm/glm.hpp>
#include <embree3/rtcore.h>

struct camera_t {
    glm::vec3 origin;
    glm::vec3 imagePlaneTopLeft;
    glm::vec3 pixelRight;
    glm::vec3 pixelDown;
};

struct material_t {
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emission;
    glm::vec3 ambient;
    float roughness;
    bool ggx;
    bool mirror;
};

struct quadLight_t {
    std::vector<glm::vec3> vertices;
    glm::vec3 color;
    glm::vec3 intensity;
    float area; 
};

struct directionalLight_t {
    glm::vec3 toLight;
    glm::vec3 brightness;
};

struct pointLight_t {
    glm::vec3 point;
    glm::vec3 brightness;
    glm::vec3 attenuation;
};

class Scene {

public:

    glm::uvec2 imageSize;
    int maxDepth;
    std::string outputFileName;
    camera_t camera;
    std::vector<glm::mat3> sphereNormalTransforms;
    std::vector<material_t> sphereMaterials;
    std::vector<material_t> triMaterials;
    std::vector<quadLight_t> quadLights;
    std::vector<directionalLight_t> directionalLights;
    std::vector<pointLight_t> pointLights;
    int lightSamples;
    int samplePerPixel;
    bool lightStratify;
    bool nee;
    bool rr;
    bool cosine; 
    bool brdf; 
    bool hemisphere; 
    bool ggx; 
    float gamma;
    bool mis;
    std::vector<unsigned char> image; 
    std::vector<float> imagef; 
    int n;
    int width;
    int height;
    std::vector<float> summedArea;
    RTCScene embreeScene;

    bool castRay(
        glm::vec3 origin,
        glm::vec3 direction,
        glm::vec3* hitPosition,
        glm::vec3* hitNormal,
        material_t* hitMaterial) const;

    bool castOcclusionRay(
        glm::vec3 origin,
        glm::vec3 direction,
        float maxDistance = std::numeric_limits<float>::infinity()) const;

    float getSummedArea(glm::vec2 topLeft, glm::vec2 bottomRight) const;
};
