#pragma once
#include <Snow/Snow.h>
#include <Snow/SnowGrid.h>
#include <Snow/Caches.h>
#include <Snow/ImplicitCollider.h>

class SnowSimulator
{
public:
	static SnowSimulator* instance();
	void update(const float dt);

	// Returns whether it actually did start
	bool start();
	void pause();
	void resume();
	void stop();
	void reset();
	bool isRunning() { return m_running; }

	void addParticleSystem(const Snow &particles);
	void clearParticleSystem();
	Snow* particleSystem() { return m_snowCollection; }

	void setGrid(const Grid &grid) { m_grid = grid; }
	Grid getGrid() {return m_grid; }

	void addCollider( const ImplicitCollider &collider ) { m_colliders += collider; }
	void addCollider(const ColliderType &t,const CUDAVec3 &center, const CUDAVec3 &param, const CUDAVec3 &velocity);

	void clearColliders() { m_colliders.clear(); }
	QVector<ImplicitCollider>& colliders() { return m_colliders; }

private:
	SnowSimulator();
	~SnowSimulator();

	void initializeCudaResources();
	void freeCudaResources();

	static SnowSimulator* m_instance;

	// CPU data structures
	Snow *m_snowCollection;
	Grid m_grid;
	QVector<ImplicitCollider> m_colliders;

	// CUDA pointers
	cudaGraphicsResource *m_particlesResource; // Particles
	cudaGraphicsResource *m_nodesResource; // Particle grid nodes
	Grid *m_devGrid;

	NodeCache *m_devNodeCaches;

	SnowParticleCache *m_hostParticleCache;
	SnowParticleCache *m_devParticleCache;

	ImplicitCollider *m_devColliders;
	SnowMaterial *m_devMaterial;

	bool m_running;
	bool m_paused;
};