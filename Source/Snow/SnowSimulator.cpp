#include "SnowSimulator.h"
#include <Snow/Cuda/Functions.h>

SnowSimulator::SnowSimulator()
	: m_snowCollection(NULL),
	  m_hostParticleCache(NULL),
	  m_running(false),
	  m_paused(false)
{
	// Add a plane collider as the ground
	CUDAVec3 center(0, 0, 0);
	CUDAVec3 param(0, 1, 0); // up normal
	CUDAVec3 velocity(0, 0, 0); // fixed
	ImplicitCollider ground(ImplicitCollider::HALF_PLANE, center, param, velocity, 0.2f);
	addCollider(ground);
}

SnowSimulator::~SnowSimulator()
{
	if (m_running) stop();
	SAFE_DELETE(m_hostParticleCache);
}

SnowSimulator* SnowSimulator::m_instance = 0;

SnowSimulator* SnowSimulator::instance()
{
	static QMutex mutex;
	if (!m_instance) 
	{
		QMutexLocker locker(&mutex);
		if (!m_instance)
			m_instance = new SnowSimulator;
	}

	return m_instance;
}

void SnowSimulator::initializeCudaResources()
{
	qDebug() << "Initializing CUDA resources...";

	// Particles
	registerVBO( &m_particlesResource, m_snowCollection->particleVBO() );
	float particlesSize = m_snowCollection->particleSize()*sizeof(SnowParticle) / 1e6;
	qDebug() <<  "Allocated"<<particlesSize<<"MB for particle system.";

	int numNodes = m_grid.nodeCount();
	int numParticles = m_snowCollection->particleSize();

	// Grid Nodes
	registerVBO( &m_nodesResource, m_snowCollection->gridVBO() );
	float nodesSize =  numNodes*sizeof(Node) / 1e6;
	qDebug() <<  "Allocating" <<  nodesSize << "MB for grid nodes.";

	// Grid
	checkCudaErrors(cudaMalloc( (void**)&m_devGrid, sizeof(Grid) ));
	checkCudaErrors(cudaMemcpy( m_devGrid, &m_grid, sizeof(Grid), cudaMemcpyHostToDevice ));

	// Colliders
	checkCudaErrors(cudaMalloc( (void**)&m_devColliders, m_colliders.size()*sizeof(ImplicitCollider) ));
	checkCudaErrors(cudaMemcpy( m_devColliders, m_colliders.data(), m_colliders.size()*sizeof(ImplicitCollider), cudaMemcpyHostToDevice ));

	// Caches
	checkCudaErrors(cudaMalloc( (void**)&m_devNodeCaches, numNodes*sizeof(NodeCache)) );
	checkCudaErrors(cudaMemset( m_devNodeCaches, 0, numNodes*sizeof(NodeCache)) );
	float nodeCachesSize = numNodes*sizeof(NodeCache) / 1e6;
	qDebug() <<  "Allocating" << nodeCachesSize << "MB for implicit update node cache.";

	SAFE_DELETE( m_hostParticleCache );
	m_hostParticleCache = new SnowParticleCache;
	cudaMalloc( (void**)&m_hostParticleCache->sigmas, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_hostParticleCache->Aps, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_hostParticleCache->FeHats, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_hostParticleCache->ReHats, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_hostParticleCache->SeHats, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_hostParticleCache->dFs, numParticles*sizeof(mat3) );
	cudaMalloc( (void**)&m_devParticleCache, sizeof(SnowParticleCache) );
	cudaMemcpy( m_devParticleCache, m_hostParticleCache, sizeof(SnowParticleCache), cudaMemcpyHostToDevice );
	float particleCachesSize = numParticles*6*sizeof(mat3) / 1e6;
	qDebug() <<  "Allocating" << particleCachesSize << "MB for implicit update particle caches.";

	qDebug() <<  "Allocated"<< particlesSize + nodesSize + nodeCachesSize + particleCachesSize << "MB in total";

	qDebug() <<  "Computing particle volumes...";
	cudaGraphicsMapResources( 1, &m_particlesResource, 0 );
	SnowParticle *devParticles;
	size_t size;
	checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devParticles, &size, m_particlesResource ) );
	if ( (int)(size/sizeof(SnowParticle)) != m_snowCollection->particleSize() )
		qWarning() <<  "SnowParticle resource error :"<<size<<"bytes ("<< m_snowCollection->particleSize()*sizeof(SnowParticle) <<"expected)";

	initializeParticleVolumes( devParticles, m_snowCollection->particleSize(), m_devGrid, numNodes );
	checkCudaErrors( cudaGraphicsUnmapResources(1, &m_particlesResource, 0) );

	qDebug() <<  "Initialization complete.";
}

void SnowSimulator::freeCudaResources()
{
	unregisterVBO( m_particlesResource );
	unregisterVBO( m_nodesResource );
	cudaFree( m_devGrid );
	cudaFree( m_devColliders );
	cudaFree( m_devNodeCaches );

	// Free the particle cache using the host structure
	cudaFree( m_hostParticleCache->sigmas );
	cudaFree( m_hostParticleCache->Aps );
	cudaFree( m_hostParticleCache->FeHats );
	cudaFree( m_hostParticleCache->ReHats );
	cudaFree( m_hostParticleCache->SeHats );
	cudaFree( m_hostParticleCache->dFs );
	SAFE_DELETE( m_hostParticleCache );
	cudaFree( m_devParticleCache );

	cudaFree( m_devMaterial );

	qDebug() << "Freed CUDA resources.";
}

void SnowSimulator::update(const float dt)
{
	if ( !m_running ) 
	{
		qDebug() << "Snow Simulation not running...";
		return;
	}
	if ( m_paused ) 
	{
		qDebug() << "Snow Simulation paused...";
		return;
	}

	cudaGraphicsMapResources( 1, &m_particlesResource, 0 );
	SnowParticle *devParticles;
	size_t size;
	checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devParticles, &size, m_particlesResource ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	if ( (int)(size/sizeof(SnowParticle)) != m_snowCollection->particleSize() )
		qWarning() <<  "SnowParticle resource error :"<<size<<"bytes ("<< m_snowCollection->particleSize()*sizeof(SnowParticle) <<"expected)";

	cudaGraphicsMapResources( 1, &m_nodesResource, 0 );
	Node *devNodes;
 	checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devNodes, &size, m_nodesResource ) );
 	checkCudaErrors( cudaDeviceSynchronize() );

	if ( (int)(size/sizeof(Node)) != m_snowCollection->gridSize() )
		qWarning() <<  "Grid nodes resource error :"<<size<<"bytes ("<< m_snowCollection->gridSize()*sizeof(Node) <<"expected)";

	updateParticles( devParticles, m_devParticleCache, m_hostParticleCache, m_snowCollection->particleSize(), m_devGrid,
		devNodes, m_devNodeCaches, m_grid.nodeCount(), m_devColliders, m_colliders.size(),
		dt, false );

	checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_particlesResource, 0 ) );
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &m_nodesResource, 0 ) );
	checkCudaErrors( cudaDeviceSynchronize() );
}

void SnowSimulator::addSnowInstance( const Snow &instance )
{
	*m_snowCollection += instance;
}

void SnowSimulator::clearAllSnowInstances()
{
	m_snowCollection->clear();
}

bool SnowSimulator::start()
{
	if (m_snowCollection->particleSize() == 0)
	{
		qWarning() << "Empty snow collection. Snow simulator starting failed.";
		return false;
	}
	if (m_grid.isEmpty())
	{
		qWarning() << "Empty snow simulation grid. Snow simulator starting failed.";
		return false;
	}
	if (m_running)
	{
		qWarning() << "Snow simulatior already running. Snow simulator starting failed.";
		return false;
	}

	initializeCudaResources();
	m_running = true;

	return true;
}

void SnowSimulator::pause()
{
	m_paused = true;
}

void SnowSimulator::resume()
{
	m_paused = false;
}

void SnowSimulator::stop()
{
	if(m_snowCollection) freeCudaResources();
	m_running = false;
}

void SnowSimulator::reset()
{
	if ( !m_running ) {
		clearColliders();
		clearAllSnowInstances();
	}
}
