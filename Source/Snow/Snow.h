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
	void reset();
	inline int size() const { return m_particles.size(); }
	inline void resize( int n ) { m_particles.resize(n); }

	SnowParticle* data() { return m_particles.data(); }
	const QVector<SnowParticle>& getParticles() const { return m_particles; }
	QVector<SnowParticle>& particles() { return m_particles; }

	GLuint vbo() { if ( !hasBuffers() ) buildBuffers(); return m_glVBO; }

	void merge( const Snow &other ) { m_particles += other.m_particles; reset(); }

	Snow& operator += ( const Snow &other ) { m_particles += other.m_particles; reset(); return *this; }
	Snow& operator += ( const SnowParticle &particle ) { m_particles.append(particle); reset(); return *this; }

private:
	bool hasBuffers() const;
	void buildBuffers();
	void deleteBuffers();

	void installShader();
	void voxelizeMesh();
	QVector<SnowParticle> m_particles;
	ParticleTechniquePtr m_renderingEffect;
	GLuint m_glVBO;
	GLuint m_glVAO;

	// Snow properties
	float m_cellSize;
	uint m_particleCount;
	float m_density;
	uint m_snowMaterial;
};

typedef QSharedPointer<Snow> SnowPtr;