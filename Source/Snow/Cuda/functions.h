#pragma once

#include <Snow/GridNode.h>

typedef unsigned int GLuint;
//struct cudaGraphicsResource;
struct Grid;
struct SnowParticle;
struct SnowParticleCache;
struct Node;
struct NodeCache;
struct ImplicitCollider;
struct SimulationParameters;
struct SnowMaterial;

extern "C"
{
	// OpenGL-CUDA interop
	void registerVBO( cudaGraphicsResource **resource, GLuint vbo );
	void unregisterVBO( cudaGraphicsResource *resource );

	// Particle simulation
	void updateParticles( SnowParticle *particles, SnowParticleCache *devParticleCache, SnowParticleCache *hostParticleCache, int numParticles,
						  Grid *grid, Node *nodes, NodeCache *nodeCache, int numNodes,
						  ImplicitCollider *colliders, int numColliders,
						  float timeStep, bool implicitUpdate );

	// Mesh filling
	void fillMesh( cudaGraphicsResource **resource, int triCount, const Grid &grid, SnowParticle *particles, int particleCount, float targetDensity, int materialPreset);


	// One time computation to get particle volumes
	void initializeParticleVolumes( SnowParticle *particles, int numParticles, const Grid *grid, int numNodes );
}