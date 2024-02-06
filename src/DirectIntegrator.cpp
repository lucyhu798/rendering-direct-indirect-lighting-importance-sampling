#include <algorithm>
#include <iostream>
#include <glm/glm.hpp>

#include "Integrator.h"

glm::vec3 DirectIntegrator::computeShading(
    glm::vec3 incidentDirection,
    glm::vec3 toLight,
    glm::vec3 normal,
    glm::vec3 lightBrightness,
    const material_t& material)
{
    return glm::vec3(0); 
}

glm::vec3 DirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction, int depth)
{
    glm::vec3 outputColor = glm::vec3(0.0f, 0.0f, 0.0f);

    glm::vec3 hitPosition;
    glm::vec3 hitNormal;
    material_t hitMaterial;

    bool hit = _scene->castRay(origin, direction, &hitPosition, &hitNormal, &hitMaterial);
    if (hit) {
        for (quadLight_t light : _scene->quadLights){
            glm::vec3 sample = glm::vec3(0);
            glm::vec3 color = glm::vec3(0);
            glm::vec3 w_i = glm::vec3(0);
            for(int i = 0; i < std::sqrt(_scene->lightSamples); i++){
                for(int j = 0; j < std::sqrt(_scene->lightSamples); j++){
                    glm::vec2 random = glm::vec2((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)));
                    if(_scene->lightStratify == true){
                        int M = std::sqrt(_scene->lightSamples); 
                        sample = light.vertices[0] + ((i + random.x)/M) * (light.vertices[1] - light.vertices[0]) + ((j + random.y)/M) * (light.vertices[3] - light.vertices[0]);
                    }
                    else{
                         sample = light.vertices[0] + random.x * (light.vertices[1] - light.vertices[0]) + random.y * (light.vertices[3] - light.vertices[0]);
                    }
                    outputColor += hitMaterial.emission;
                    w_i = normalize(sample - hitPosition);
                    bool occluded = _scene->castOcclusionRay(sample, -w_i, glm::length(sample - hitPosition));
                    if(!occluded) {
                        glm::vec3 reflect = glm::reflect(direction, hitNormal);

                        glm::vec3 f_xa = hitMaterial.diffuse * (float)(1/M_PI);
                        float f_xb = (hitMaterial.shininess + 2)/(2 * M_PI);
                        float f_xc = glm::pow(glm::dot(reflect, w_i), hitMaterial.shininess);
                        glm::vec3 f_x = f_xa + (hitMaterial.specular * f_xb) * f_xc;

                        glm::vec3 quadNormal = glm::normalize(glm::cross(light.vertices[1] - light.vertices[0], light.vertices[3] - light.vertices[0]));

                        float g_xa = (1.0f / glm::pow(glm::length(sample - hitPosition), 2));
                        float g_xb = glm::dot(hitNormal, glm::normalize(sample - hitPosition));
                        float g_xc = glm::dot(quadNormal, glm::normalize(sample - hitPosition));
                        float g_x = g_xa * g_xb * g_xc;
                        
                        color += g_x * f_x;
                    }
                }
            }
            outputColor += color * light.intensity * light.area / (float) _scene->lightSamples;   
        }   
    }

    return outputColor;
}

glm::vec3 DirectIntegrator::traceRay(glm::vec3 origin, glm::vec3 direction)
{
    return traceRay(origin, direction, 0);
}

