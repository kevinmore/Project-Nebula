#include <limits.h>
#include <assert.h>
#include "ShadingTechnique.h"
#include <QtMath>
#include <Scene/Scene.h>

ShadingTechnique::ShadingTechnique(const QString &shaderName, ShaderType shaderType, Scene* scene)
	: Technique(shaderName),
	  m_scene(scene),
	  m_shaderType(shaderType),
	  usingCubeMap(false)
{
	if (!init()) 
	{
		qWarning() << shaderName << "may not be initialized successfully.";
	}
}

void ShadingTechnique::enable()
{
	m_shaderProgram->bind();
	if (usingCubeMap && m_scene)
	{
		SkyboxPtr skybox = m_scene->getSkybox();
		if(!skybox) return;
		CubemapTexturePtr cubeMap = skybox->getCubemapTexture();
		cubeMap->bind(GL_TEXTURE0);
	}
}

bool ShadingTechnique::init()
{
    if (!Technique::init() || m_shaderFileName.isEmpty()) 
	{
        return false;
    }

	return compileShader();
}

bool ShadingTechnique::compileShader()
{
	QString vertexShader = (m_shaderType == STATIC) ? "static.vert" : "skinning.vert";
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + vertexShader);
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + m_shaderFileName + ".frag");
	m_shaderProgram->link();

	if (m_shaderFileName.contains("reflection"))
		usingCubeMap = true;
	else
		usingCubeMap = false;

	PointLight pl[2];
	pl[0].Position = vec3(2,2,0);
	pl[0].Color = vec3(1.0f, 1.0f, 1.0f);
	pl[0].AmbientIntensity = 0.55f;
	pl[0].DiffuseIntensity = 0.9f;

	pl[1].Position = vec3(-2,2,0);
	pl[1].Color = vec3(1.0f, 1.0f, 1.0f);
	pl[1].AmbientIntensity = 0.55f;
	pl[1].DiffuseIntensity = 0.9f;

	enable();
	setPointLights(2, pl);

	initLights();

	return true;
}

void ShadingTechnique::setDirectionalLight(const DirectionalLight& Light)
{
   	m_shaderProgram->setUniformValue("gDirectionalLight.Base.Color", Light.Color);
 	m_shaderProgram->setUniformValue("gDirectionalLight.Direction", Light.Direction.normalized());
 	m_shaderProgram->setUniformValue("gDirectionalLight.Base.AmbientIntensity", Light.AmbientIntensity);
 	m_shaderProgram->setUniformValue("gDirectionalLight.Base.DiffuseIntensity", Light.DiffuseIntensity);
}


void ShadingTechnique::setMVPMatrix(const mat4& mvp)
{
	m_shaderProgram->setUniformValue("gWVP", mvp);
}

void ShadingTechnique::setModelMatrix(const mat4& model)
{
	m_shaderProgram->setUniformValue("gWorld", model);
}


void ShadingTechnique::setViewMatrix( const mat4& view )
{
	m_shaderProgram->setUniformValue("viewMatrix", view);
}

void ShadingTechnique::setColorTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gColorMap", TextureUnit);
}

void ShadingTechnique::setShadowMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gShadowMap", TextureUnit);
}

void ShadingTechnique::setNormalMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gNormalMap", TextureUnit);
}

void ShadingTechnique::setEyeWorldPos(const vec3& EyeWorldPos)
{
	m_shaderProgram->setUniformValue("gEyeWorldPos", EyeWorldPos);
}

void ShadingTechnique::setPointLights(unsigned int NumLights, const PointLight* pLights)
{
	m_shaderProgram->setUniformValue("gNumPointLights", NumLights);
	for(uint i = 0; i < NumLights; ++i)
	{
		QString lightString = "gPointLights["+ QString::number(i) + "].";

		m_shaderProgram->setUniformValue((lightString + "Base.Color").toStdString().c_str(), pLights[i].Color);
		m_shaderProgram->setUniformValue((lightString + "Base.AmbientIntensity").toStdString().c_str(), pLights[i].AmbientIntensity);
		m_shaderProgram->setUniformValue((lightString + "Base.DiffuseIntensity").toStdString().c_str(), pLights[i].DiffuseIntensity);
		m_shaderProgram->setUniformValue((lightString + "Position").toStdString().c_str(), pLights[i].Position);
		m_shaderProgram->setUniformValue((lightString + "Atten.Constant").toStdString().c_str(), pLights[i].Attenuation.Constant);
		m_shaderProgram->setUniformValue((lightString + "Atten.Linear").toStdString().c_str(), pLights[i].Attenuation.Linear);
		m_shaderProgram->setUniformValue((lightString + "Atten.Exp").toStdString().c_str(), pLights[i].Attenuation.Exp);
	}

}

void ShadingTechnique::setSpotLights(unsigned int NumLights, const SpotLight* pLights)
{

	m_shaderProgram->setUniformValue("gNumSpotLights", NumLights);
	for(uint i = 0; i < NumLights; ++i)
	{
		QString lightString = "gSpotLights["+ QString::number(i) + "].";

		m_shaderProgram->setUniformValue((lightString + "Base.Base.Color").toStdString().c_str(), pLights[i].Color);
		m_shaderProgram->setUniformValue((lightString + "Base.Base.AmbientIntensity").toStdString().c_str(), pLights[i].AmbientIntensity);
		m_shaderProgram->setUniformValue((lightString + "Base.Base.DiffuseIntensity").toStdString().c_str(), pLights[i].DiffuseIntensity);
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Constant").toStdString().c_str(), pLights[i].Attenuation.Constant);
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Linear").toStdString().c_str(), pLights[i].Attenuation.Linear);
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Exp").toStdString().c_str(), pLights[i].Attenuation.Exp);
		m_shaderProgram->setUniformValue((lightString + "Position").toStdString().c_str(), pLights[i].Position);
		m_shaderProgram->setUniformValue((lightString + "Direction").toStdString().c_str(), pLights[i].Direction.normalized());
		m_shaderProgram->setUniformValue((lightString + "Cutoff").toStdString().c_str(), cosf(qDegreesToRadians(pLights[i].Cutoff)));
	}
}


void ShadingTechnique::setBoneTransform(uint Index, const mat4& Transform)
{
    assert(Index < MAX_BONES);
	QString boneString = "gBones["+ QString::number(Index) + "]";
	m_shaderProgram->setUniformValue(boneString.toStdString().c_str(), Transform);
}

void ShadingTechnique::initLights()
{
	if (!m_scene) return;
	QList<LightPtr> lights = m_scene->getLights();

	qDebug() << "Lights count:" << lights.size();
}

void ShadingTechnique::setMaterial( const Material& mat )
{
	m_shaderProgram->setUniformValue("material.Ka", mat.m_ambientColor);
	m_shaderProgram->setUniformValue("material.Kd", mat.m_diffuseColor);
	m_shaderProgram->setUniformValue("material.Ks", mat.m_specularColor);
	m_shaderProgram->setUniformValue("material.Ke", mat.m_emissiveColor);
	m_shaderProgram->setUniformValue("material.shininessStrength", mat.m_shininessStrength);
	m_shaderProgram->setUniformValue("material.shininess", mat.m_shininess);
	m_shaderProgram->setUniformValue("material.roughnessValue", mat.m_roughness);
	m_shaderProgram->setUniformValue("material.fresnelReflectance", mat.m_fresnelReflectance);
}

void ShadingTechnique::setMaterial( const MaterialPtr mat )
{
	m_shaderProgram->setUniformValue("material.Ka", mat->m_ambientColor);
	m_shaderProgram->setUniformValue("material.Kd", mat->m_diffuseColor);
	m_shaderProgram->setUniformValue("material.Ks", mat->m_specularColor);
	m_shaderProgram->setUniformValue("material.Ke", mat->m_emissiveColor);
	m_shaderProgram->setUniformValue("material.shininessStrength", mat->m_shininessStrength);
	m_shaderProgram->setUniformValue("material.shininess", mat->m_shininess);
	m_shaderProgram->setUniformValue("material.roughnessValue", mat->m_roughness);
	m_shaderProgram->setUniformValue("material.fresnelReflectance", mat->m_fresnelReflectance);
}


void ShadingTechnique::setMatAmbientColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ka", col);
}

void ShadingTechnique::setMatDiffuseColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Kd", col);
}

void ShadingTechnique::setMatSpecularColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ks", col);
}

void ShadingTechnique::setMatEmissiveColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ke", col);
}

void ShadingTechnique::setMatSpecularIntensity(float intensity)
{
	m_shaderProgram->setUniformValue("material.shininessStrength", intensity);
}

void ShadingTechnique::setMatSpecularPower(float power)
{
	m_shaderProgram->setUniformValue("material.shininess", power);
}

void ShadingTechnique::setMatRoughnessValue( float val )
{
	m_shaderProgram->setUniformValue("material.roughnessValue", val);
}

void ShadingTechnique::setMatFresnelReflectance( float val )
{
	m_shaderProgram->setUniformValue("material.fresnelReflectance", val);
}
