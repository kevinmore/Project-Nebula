#include "ParticleSystem.h"
#include <Scene/Scene.h>
#include <Utility/Math.h>

ParticleSystem::ParticleSystem(Scene* scene)
	: Component(1),// get rendered last
	  m_curReadBufferIndex(0),
	  m_aliveParticlesCount(0),
	  m_elapsedTime(0.0f),
	  m_nextEmitTime(0.0f),
	  m_scene(scene),
	  m_initialized(false),
	  m_generateImpulse(false)
{}

ParticleSystem::~ParticleSystem()
{}

void ParticleSystem::initParticleSystem()
{
	Q_ASSERT(initializeOpenGLFunctions());	

	installShaders();

	prepareTransformFeedback();

	// load texture
	loadTexture("../Resource/Textures/flares/nova.png");

	// reset the properties to default
	resetEmitter();
}


void ParticleSystem::installShaders()
{
	// Updating program
	particleUpdater = ParticleTechniquePtr(new ParticleTechnique("particles_update", ParticleTechnique::UPDATE));
	if (!particleUpdater->init()) 
	{
		qWarning() << "particles_update initializing failed.";
		return;
	}

	// Rendering program
	particleRenderer = ParticleTechniquePtr(new ParticleTechnique("particles_render", ParticleTechnique::RENDER));
	if (!particleRenderer->init()) 
	{
		qWarning() << "particles_render initializing failed.";
		return;
	}

	m_initialized = true;
}

void ParticleSystem::prepareTransformFeedback()
{
	if(!m_initialized) return;

	glGenTransformFeedbacks(1, &m_transformFeedbackBuffer);
	glGenQueries(1, &m_query);

	glGenVertexArrays(2, m_VAO);
	glGenBuffers(2, m_particleBuffer);

	Particle firstParticle;
	firstParticle.iType = PARTICLE_TYPE_GENERATOR;
	m_aliveParticlesCount = 1;

	for (int i = 0; i < 2; ++i)
	{
		glBindVertexArray(m_VAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, m_particleBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Particle) * MAX_PARTICLES_PER_SYSTEM, NULL, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Particle), &firstParticle);

		for (int j = 0; j < NUM_PARTICLE_ATTRIBUTES; ++j)
		{
			glEnableVertexAttribArray(j);
		}


		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*)0); // Position
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*)12); // Velocity
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*)24); // Color
		glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*)36); // Lifetime
		glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (const GLvoid*)40); // Size
		glVertexAttribPointer(5, 1, GL_INT,	  GL_FALSE, sizeof(Particle), (const GLvoid*)44); // Type
	}

	particleUpdater->setVAO(m_VAO[0]);
	particleRenderer->setVAO(m_VAO[1]);
}


void ParticleSystem::updateParticles( float fTimePassed )
{
	particleUpdater->enable();

	QQuaternion rotation;
	vec3 scale;
	Math::decomposeMat4(m_actor->getTransformMatrix(), scale, rotation, m_position);

	particleUpdater->getShaderProgram()->setUniformValue("fTimePassed", fTimePassed);
	particleUpdater->getShaderProgram()->setUniformValue("fParticleMass", m_particleMass);
	particleUpdater->getShaderProgram()->setUniformValue("fGravityFactor", m_gravityFactor);
	particleUpdater->getShaderProgram()->setUniformValue("vGenPosition", m_position);
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityMin", rotation.rotatedVector(m_minVelocity));
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityRange", rotation.rotatedVector(m_velocityRange));
	particleUpdater->getShaderProgram()->setUniformValue("vForce", m_force);

	if (m_isCollisionEnabled)
	{
		particleUpdater->getShaderProgram()->setUniformValue("iCollisionEnabled", 1);
		if(m_collider)
		{
			vec3 rot = m_collider->rotation();
			QQuaternion rotation = QQuaternion::fromAxisAndAngle(Math::Vector3::UNIT_X, rot.x())
				* QQuaternion::fromAxisAndAngle(Math::Vector3::UNIT_Y, rot.y())
				* QQuaternion::fromAxisAndAngle(Math::Vector3::UNIT_Z, rot.z());

			m_planeNormal = rotation.rotatedVector(Math::Vector3::UNIT_Y);
			m_planePoint = m_collider->position();
		}
		particleUpdater->getShaderProgram()->setUniformValue("vPlaneNormal", m_planeNormal);
		particleUpdater->getShaderProgram()->setUniformValue("vPlanePoint", m_planePoint);
		particleUpdater->getShaderProgram()->setUniformValue("fRestitution", m_restitution);
	}
	else
		particleUpdater->getShaderProgram()->setUniformValue("iCollisionEnabled", 0);

	if (m_isColorRandomed)
		particleUpdater->getShaderProgram()->setUniformValue("vGenColor", Math::Random::randUnitVec3());
	else
		particleUpdater->getShaderProgram()->setUniformValue("vGenColor", m_color);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_minLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", m_lifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_minLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", m_lifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenSize", m_size);
	particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", 0);

	if (m_elapsedTime > m_nextEmitTime)
	{
		particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", m_emitAmount);

		vec3 vRandomSeed =  Math::Random::randUnitVec3();
		particleUpdater->getShaderProgram()->setUniformValue("vRandomSeed", vRandomSeed);
		m_nextEmitTime = m_elapsedTime + m_emitRate;
	}

	glEnable(GL_RASTERIZER_DISCARD);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_transformFeedbackBuffer);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glEnableVertexAttribArray(1); // Re-enable velocity

	// store the results of transform feedback
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_particleBuffer[1-m_curReadBufferIndex]);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, m_query);
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, m_aliveParticlesCount);

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	glGetQueryObjectiv(m_query, GL_QUERY_RESULT, &m_aliveParticlesCount);

	m_curReadBufferIndex = 1-m_curReadBufferIndex;

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
	
	if(m_generateImpulse)
	{
		// calculate the linear impulse this particle system generated
		float totalMass = m_particleMass * m_emitAmount;
		//vec3 avgVel = Math::Random::random(m_minVelocity, m_maxVelocity);
		vec3 avgVel = -(rotation.rotatedVector(m_minVelocity) + rotation.rotatedVector(m_maxVelocity)) * 0.5f;
		m_vLinearImpulse = totalMass * avgVel * 0.01f;
	}
}

void ParticleSystem::render(const float currentTime)
{
	if(!m_initialized) return;
	float dt = currentTime - m_elapsedTime;
	m_elapsedTime = currentTime;


	updateParticles(dt);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glDepthMask(0);
	
	glDisable(GL_RASTERIZER_DISCARD);
	particleRenderer->enable();
	m_Texture->bind(DIFFUSE_TEXTURE_UNIT);

	mat4 matProjection = m_scene->getCamera()->projectionMatrix();
	mat4 matView = m_scene->getCamera()->viewMatrix();
	vQuad1 = vec3::crossProduct(m_scene->getCamera()->viewVector(), m_scene->getCamera()->upVector()).normalized();
	vQuad2 = vec3::crossProduct(m_scene->getCamera()->viewVector(), vQuad1).normalized();

	particleRenderer->getShaderProgram()->setUniformValue("matrices.mProj", matProjection);
	particleRenderer->getShaderProgram()->setUniformValue("matrices.mView", matView);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad1", vQuad1);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad2", vQuad2);
	particleRenderer->getShaderProgram()->setUniformValue("gSampler", DIFFUSE_TEXTURE_UNIT);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glDisableVertexAttribArray(1); // Disable velocity, because we don't need it for rendering

	glDrawArrays(GL_POINTS, 0, m_aliveParticlesCount);
	glDepthMask(1);	
	glDisable(GL_BLEND);
}

QColor ParticleSystem::getParticleColor() const
{
	QColor col;
	col.setRgbF(m_color.x(), m_color.y(), m_color.z());
	return col;
}

void ParticleSystem::setParticleColor( const QColor& col )
{
	m_color.setX(col.redF()); 
	m_color.setY(col.greenF()); 
	m_color.setZ(col.blueF());
}

void ParticleSystem::loadTexture( const QString& fileName )
{
	// clear the previous loaded texture
	m_scene->textureManager()->deleteTexture(m_actor->objectName() + "_texture");

	// load a new one
	m_Texture = m_scene->textureManager()->addTexture(m_actor->objectName() + "_texture", fileName);
}

QString ParticleSystem::getTextureFileName()
{
	// extract the relative path
	QDir dir;
	return dir.relativeFilePath(m_Texture->fileName());
}

void ParticleSystem::resetEmitter()
{
	m_particleMass = 0.015f;
	m_gravityFactor = 0.01f;
	
	m_minVelocity = vec3(-0.2f, 0.2f, -0.2f);
	m_maxVelocity = vec3(0.2f, 0.3f, 0.2f);
	m_velocityRange = m_maxVelocity - m_minVelocity;
	
	m_force = Math::Vector3::ZERO;
	m_isCollisionEnabled = false;
	m_planePoint = Math::Vector3::ZERO;
	m_planeNormal = Math::Vector3::UNIT_Y;
	m_restitution = 1.0f;

	m_isColorRandomed = false;
	m_color = vec3(0, 0.5, 1);
	m_size = 0.02f;
	
	m_minLife = 8.0f;
	m_maxLife = 10.0f;
	m_lifeRange = m_maxLife - m_minLife;
	
	m_emitRate = 0.02f;
	
	m_emitAmount = 30;
}

void ParticleSystem::assingCollisionObject( GameObjectPtr collider )
{
	m_collider = collider;
}
