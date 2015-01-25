#include "ParticleSystem.h"
#include <Scene/Scene.h>
#include <Utility/Math.h>

ParticleSystem::ParticleSystem(Scene* scene)
	: Component(1),// get rendered last
	  m_curReadBufferIndex(0),
	  m_aliveParticlesCount(0),
	  fElapsedTime(0.0f),
	  fNextEmitTime(0.0f),
	  m_scene(scene),
	  bInitialized(false)
{
}

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

	bInitialized = true;
}

void ParticleSystem::prepareTransformFeedback()
{
	if(!bInitialized) return;

	glGenTransformFeedbacks(1, &m_transformFeedbackBuffer);
	glGenQueries(1, &uiQuery);

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
	Math::decomposeMat4(m_actor->modelMatrix(), scale, rotation, vGenPosition);

	particleUpdater->getShaderProgram()->setUniformValue("fTimePassed", fTimePassed);
	particleUpdater->getShaderProgram()->setUniformValue("fParticleMass", m_fParticleMass);
	particleUpdater->getShaderProgram()->setUniformValue("fGravityFactor", m_fGravityFactor);
	particleUpdater->getShaderProgram()->setUniformValue("vGenPosition", vGenPosition);
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityMin", rotation.rotatedVector(m_minVelocity));
	particleUpdater->getShaderProgram()->setUniformValue("vGenVelocityRange", rotation.rotatedVector(vGenVelocityRange));
	particleUpdater->getShaderProgram()->setUniformValue("vForce", m_force);

	if (bCollisionEnabled)
	{
		particleUpdater->getShaderProgram()->setUniformValue("iCollisionEnabled", 1);
		if(m_collider)
		{
			vec3 rot = m_collider->rotation();
			QQuaternion rotation = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_X, rot.x())
				* QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Y, rot.y())
				* QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Z, rot.z());

			vPlaneNormal = rotation.rotatedVector(Math::Vector3D::UNIT_Y);
			vPlanePoint = m_collider->position();
		}
		particleUpdater->getShaderProgram()->setUniformValue("vPlaneNormal", vPlaneNormal);
		particleUpdater->getShaderProgram()->setUniformValue("vPlanePoint", vPlanePoint);
		particleUpdater->getShaderProgram()->setUniformValue("fRestitution", fRestitution);
	}
	else
		particleUpdater->getShaderProgram()->setUniformValue("iCollisionEnabled", 0);

	if (bRandomColor)
		particleUpdater->getShaderProgram()->setUniformValue("vGenColor", Math::Random::randUnitVec3());
	else
		particleUpdater->getShaderProgram()->setUniformValue("vGenColor", vGenColor);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_fMinLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeMin", m_fMinLife);
	particleUpdater->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	particleUpdater->getShaderProgram()->setUniformValue("fGenSize", m_fGenSize);
	particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", 0);

	if (fElapsedTime > fNextEmitTime)
	{
		particleUpdater->getShaderProgram()->setUniformValue("iNumToGenerate", m_EmitAmount);

		vec3 vRandomSeed =  Math::Random::randUnitVec3();
		particleUpdater->getShaderProgram()->setUniformValue("vRandomSeed", vRandomSeed);
		fNextEmitTime = fElapsedTime + m_fEmitRate;
	}

	glEnable(GL_RASTERIZER_DISCARD);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_transformFeedbackBuffer);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glEnableVertexAttribArray(1); // Re-enable velocity

	// store the results of transform feedback
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_particleBuffer[1-m_curReadBufferIndex]);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, uiQuery);
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, m_aliveParticlesCount);

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	glGetQueryObjectiv(uiQuery, GL_QUERY_RESULT, &m_aliveParticlesCount);

	m_curReadBufferIndex = 1-m_curReadBufferIndex;

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
	
	// calculate the linear impulse this particle system generated
	float totalMass = m_fParticleMass * m_EmitAmount;
	//vec3 avgVel = Math::Random::random(m_minVelocity, m_maxVelocity);
	vec3 avgVel = -(rotation.rotatedVector(m_minVelocity) + rotation.rotatedVector(m_maxVelocity)) * 0.5f;
	m_vLinearImpulse = totalMass * avgVel * 0.01f;
}

void ParticleSystem::render(const float currentTime)
{
	if(!bInitialized) return;
	float dt = currentTime - fElapsedTime;
	fElapsedTime = currentTime;


	updateParticles(dt);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glDepthMask(0);
	
	glDisable(GL_RASTERIZER_DISCARD);
	particleRenderer->enable();
	m_Texture->bind(COLOR_TEXTURE_UNIT);

	mat4 matProjection = m_scene->getCamera()->projectionMatrix();
	mat4 matView = m_scene->getCamera()->viewMatrix();
	vQuad1 = vec3::crossProduct(m_scene->getCamera()->viewVector(), m_scene->getCamera()->upVector()).normalized();
	vQuad2 = vec3::crossProduct(m_scene->getCamera()->viewVector(), vQuad1).normalized();

	particleRenderer->getShaderProgram()->setUniformValue("matrices.mProj", matProjection);
	particleRenderer->getShaderProgram()->setUniformValue("matrices.mView", matView);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad1", vQuad1);
	particleRenderer->getShaderProgram()->setUniformValue("vQuad2", vQuad2);
	particleRenderer->getShaderProgram()->setUniformValue("gSampler", COLOR_TEXTURE_UNIT);

	glBindVertexArray(m_VAO[m_curReadBufferIndex]);
	glDisableVertexAttribArray(1); // Disable velocity, because we don't need it for rendering

	glDrawArrays(GL_POINTS, 0, m_aliveParticlesCount);
	glDepthMask(1);	
	glDisable(GL_BLEND);
}


// void ParticleSystem::setEmitterProperties( float particleMass, vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float emitRate, int a_iNumToGenerate )
// {
// 	m_fParticleMass = particleMass;
// 	m_fGravityFactor = 1.0f;
// 
// 	m_minVelocity = a_vGenVelocityMin;
// 	m_maxVelocity = a_vGenVelocityMax;
// 	vGenVelocityRange = m_maxVelocity - m_minVelocity;
// 
// 	m_force = a_vGenGravityVector;
// 	bRandomColor = false;
// 	vGenColor = a_vGenColor;
// 	m_fGenSize = a_fGenSize;
// 
// 	m_fMinLife = a_fGenLifeMin;
// 	m_fMaxLife = a_fGenLifeMax;
// 	fGenLifeRange = m_fMaxLife - m_fMinLife;
// 
// 	m_fEmitRate = emitRate;
// 
// 	m_EmitAmount = a_iNumToGenerate;
// }


QColor ParticleSystem::getParticleColor() const
{
	QColor col;
	col.setRedF(vGenColor.x());
	col.setGreenF(vGenColor.y());
	col.setBlueF(vGenColor.z());

	return col;
}

void ParticleSystem::setParticleColor( const QColor& col )
{
	vGenColor.setX(col.redF()); 
	vGenColor.setY(col.greenF()); 
	vGenColor.setZ(col.blueF());
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
	m_fParticleMass = 0.01f;
	m_fGravityFactor = 1.0f;
	
	m_minVelocity = vec3(-20, 20, -20);
	m_maxVelocity = vec3(20, 30, 20);
	vGenVelocityRange = m_maxVelocity - m_minVelocity;
	
	m_force = Math::Vector3D::ZERO;
	bCollisionEnabled = false;
	vPlanePoint = Math::Vector3D::ZERO;
	vPlaneNormal = Math::Vector3D::UNIT_Y;
	fRestitution = 1.0f;

	bRandomColor = false;
	vGenColor = vec3(0, 0.5, 1);
	m_fGenSize = 0.75f;
	
	m_fMinLife = 8.0f;
	m_fMaxLife = 10.0f;
	fGenLifeRange = m_fMaxLife - m_fMinLife;
	
	m_fEmitRate = 0.02f;
	
	m_EmitAmount = 30;
}

void ParticleSystem::assingCollisionObject( GameObjectPtr collider )
{
	m_collider = collider;
}
