#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <glm/glm.hpp>

#include "Scene.h"
#include "Integrator.h"

#include "RenderPool.h"

RenderJob::RenderJob(glm::uvec2 startPixel, glm::uvec2 windowSize)
    : startPixel(startPixel),
      windowSize(windowSize),
      _result(windowSize.x * windowSize.y)
{
}

void RenderJob::render(Scene* scene, Integrator* integrator)
{
    for (size_t wy = 0; wy < windowSize.y; wy++) {
        size_t y = startPixel.y + wy;
        for (size_t wx = 0; wx < windowSize.x; wx++) {
            size_t x = startPixel.x + wx;
            for(int i = 0; i < scene->samplePerPixel; i++){
                glm::vec3 target;
                if(i == 0){
                    target =
                    scene->camera.imagePlaneTopLeft
                    + (x + 0.5f) * scene->camera.pixelRight
                    + (y + 0.5f) * scene->camera.pixelDown;
                }
                else{
                    glm::vec2 random = glm::vec2((float) rand() / ((RAND_MAX + 1u)), (float)rand() / ((RAND_MAX + 1u)));
                    target =
                    scene->camera.imagePlaneTopLeft
                    + (x + random.x) * scene->camera.pixelRight
                    + (y + random.y) * scene->camera.pixelDown;

                }
                glm::vec3 direction = glm::normalize(target - scene->camera.origin);
                 _result[wy * windowSize.x + wx] += integrator->traceRay(scene->camera.origin, direction)/ (float) scene->samplePerPixel;
                 if(i == scene->samplePerPixel - 1){
                     glm::vec3 final = glm::vec3(std::pow(_result[wy * windowSize.x + wx].x, 1/scene->gamma),std::pow(_result[wy * windowSize.x + wx].y,1/scene->gamma), std::pow(_result[wy * windowSize.x + wx].z, 1/scene->gamma));
                     _result[wy * windowSize.x + wx] = final;
                 }
            }
        }
    }
}

std::vector<glm::vec3> RenderJob::getResult()
{
    return std::move(_result);
}

RenderPool::RenderPool(Scene* scene, Integrator* integrator, int numThreads, std::vector<RenderJob*>& jobs)
    : _scene(scene), _integrator(integrator), _nextJob(0), _jobQueue(jobs)
{
    for (int i = 0; i < numThreads; i++) {
        _threads.push_back(std::thread(threadMain, this));
    }
}

RenderPool::~RenderPool()
{
    for (std::thread& thread : _threads) {
        thread.join();
    }
}

void RenderPool::getCompletedJobs(std::vector<RenderJob*>& completedJobs)
{
    {
        std::unique_lock<std::mutex> lock(_mutex);

        _condition.wait(lock, [this]{ return _completedJobs.size() > 0; });
        completedJobs = std::move(_completedJobs);
    }
}

void RenderPool::threadMain(RenderPool* pool)
{
    while (true) {

        size_t jobIndex;
        {
            std::unique_lock<std::mutex> lock(pool->_mutex);

            if (pool->_nextJob >= pool->_jobQueue.size()) break;

            jobIndex = pool->_nextJob;
            pool->_nextJob++;
        }

        pool->_jobQueue[jobIndex]->render(pool->_scene, pool->_integrator);

        {
            std::unique_lock<std::mutex> lock(pool->_mutex);

            pool->_completedJobs.push_back(pool->_jobQueue[jobIndex]);
            pool->_condition.notify_all();
        }
    }
}

