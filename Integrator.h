
   
#pragma once

#include <glm/glm.hpp>

#include "Scene.h"

class Integrator {

protected:

    Scene* _scene;

public:

    void setScene(Scene* scene)
    {
        _scene = scene;
    }

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction) = 0;

};


class RayTracerIntegrator : public Integrator {

private:

    glm::vec3 computeShading(
        glm::vec3 incidentDirection,
        glm::vec3 toLight,
        glm::vec3 normal,
        glm::vec3 lightBrightness,
        const material_t& material);

    glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);

public:

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);

};
class PathTracerIntegrator : public Integrator {

private:
    glm::vec3 alignZ(glm::vec3 v, glm::vec3 normal);
    float pdf_nee(const material_t& hitMaterial, glm::vec3 wi, glm::vec3 wo, glm::vec3 n, glm::vec3 position);
    float pdf_brdf(const material_t& hitMaterial, glm::vec3 wi, glm::vec3 wo, glm::vec3 n);
    float computeG(glm::vec3 v, glm::vec3 n, float roughness);
    glm::vec3 computeBRDF(const material_t& material, glm::vec3 wi, glm::vec3 wo, glm::vec3 n);
    glm::vec3 sampleBRDF(glm::vec3 direction, glm::vec3 hitPosition, glm::vec3 hitNormal, const material_t& hitMaterial, const quadLight_t& light, glm::vec3 throughput, int depth);
    glm::vec3 sampleQuadLight(glm::vec3 direction, glm::vec3 hitPosition, glm::vec3 hitNormal, const material_t& hitMaterial, const quadLight_t& light, glm::vec3 sampleDirection);
    glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth, glm::vec3 throughput);

public:

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);

};
class DirectIntegrator : public Integrator {

private:

    glm::vec3 computeShading(
        glm::vec3 incidentDirection,
        glm::vec3 toLight,
        glm::vec3 normal,
        glm::vec3 lightBrightness,
        const material_t& material);

    glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);

public:

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);

};
class AnalyticDirectIntegrator : public Integrator {

private:

    glm::vec3 computeShading(
        glm::vec3 incidentDirection,
        glm::vec3 toLight,
        glm::vec3 normal,
        glm::vec3 lightBrightness,
        const material_t& material);

    glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction, int depth);

public:

    virtual glm::vec3 traceRay(glm::vec3 origin, glm::vec3 direction);

};

class EVPartitionNode {
    public:
        glm::vec2 topLeft;
        glm::vec2 bottomRight;
        EVPartitionNode * topLeftChild;
        EVPartitionNode * topRightChild;
        EVPartitionNode * bottomLeftChild;
        EVPartitionNode * bottomRightChild; 
        float weight;
        float area;

        EVPartitionNode(glm::vec2 topLeft, glm::vec2 bottomRight, float weight = 0.0f) {
            this->topLeft = topLeft;
            this->bottomRight = bottomRight;
            this->weight = weight;
            this->area = (bottomRight.x - topLeft.x) * (topLeft.y - bottomRight.y);
            this->topLeftChild = NULL;
            this->topRightChild = NULL;
            this->bottomLeftChild = NULL;
            this->bottomRightChild = NULL;
        }
};