#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Primitives/Component.h>
#include <QVector>
#include <QSharedPointer>
#include <Scene/ShadingTechniques/ParticleTechnique.h>
#include "SnowParticle.h"
#include "GridNode.h"

class Snow : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	Snow();
	Snow(uint particleCount);
	Snow(float cellSize, uint particleCount, float density, uint snowMaterial);
	~Snow();

	virtual QString className() { return "Snow"; }
	virtual void render(const float currentTime);

	void initialize();
	void clear();
	void resetParticleBuffers();
	void resetGridBuffers();
	inline int particleSize() const { return m_particles.size(); }
	inline int gridSize() const { return m_gridSize; }
	inline void resize( int n ) { m_particles.resize(n); }

	SnowParticle* data() { return m_particles.data(); }
	const QVector<SnowParticle>& getParticles() const { return m_particles; }
	QVector<SnowParticle>& particles() { return m_particles; }

	void setGrid( const Grid &grid );
	Grid getGrid() const { return m_grid; }

	GLuint particleVBO() { if ( !hasParticleBuffers() ) buildParticleBuffers(); return m_particleVBO; }
	GLuint gridVBO() { if ( !hasGridBuffers() ) buildGridBuffers(); return m_gridVBO; }

	void merge( const Snow &other ) { m_particles += other.m_particles; resetParticleBuffers(); }

	Snow& operator += ( const Snow &other );
	Snow& operator += ( const SnowParticle &particle );

private:
	bool hasParticleBuffers() const;
	void buildParticleBuffers();
	void deleteParticleBuffers();

	bool hasGridBuffers() const;
	void buildGridBuffers();
	void deleteGridBuffers();

	void installShader();
	void voxelizeMesh();
	QVector<SnowParticle> m_particles;
	ParticleTechniquePtr m_renderingEffect;
	GLuint m_particleVBO, m_gridVBO;
	GLuint m_VAO;

	// Snow properties
	float m_cellSize;
	uint m_particleCount;
	float m_density;
	uint m_snowMaterial;

	// Grid properties
	Grid m_grid;
	int m_gridSize;
};

typedef QSharedPointer<Snow> SnowPtr;