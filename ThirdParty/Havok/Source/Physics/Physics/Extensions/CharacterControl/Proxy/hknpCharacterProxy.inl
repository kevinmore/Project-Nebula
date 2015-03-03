/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>


HK_FORCE_INLINE const hkArray<hknpCollisionResult>& hknpCharacterProxy::getManifold() const
{
	return m_manifold;
}

HK_FORCE_INLINE const hkVector4& hknpCharacterProxy::getPosition() const
{
	return m_transform.getTranslation();
}

HK_FORCE_INLINE void hknpCharacterProxy::setPosition(hkVector4Parameter position)
{
	m_transform.setTranslation(position);

	
	//m_shapePhantom->setPosition(position, m_keepDistance + m_keepContactTolerance);
}

HK_FORCE_INLINE const hkVector4& hknpCharacterProxy::getLinearVelocity() const
{
	return m_velocity;
}

HK_FORCE_INLINE void hknpCharacterProxy::setLinearVelocity(hkVector4Parameter vel)
{
	m_velocity = vel;
}

HK_FORCE_INLINE const hkTransform& hknpCharacterProxy::getTransform() const
{
	return m_transform;
}

HK_FORCE_INLINE void hknpCharacterProxy::getAabb(hkAabb& aabb) const
{
	hkAabbUtil::calcAabb(m_transform, m_aabb, aabb);
}

HK_FORCE_INLINE const hknpWorld* hknpCharacterProxy::getWorld() const
{
	return m_world;
}

HK_FORCE_INLINE const hknpShape* hknpCharacterProxy::getShape() const
{
	return m_shape;
}

HK_FORCE_INLINE hknpBodyId hknpCharacterProxy::getPhantom() const
{
	return m_bodyId;
}

HK_FORCE_INLINE hkReal hknpCharacterProxy::getMaxSlope() const
{
	return hkMath::acos(m_maxSlopeCosine);
}

HK_FORCE_INLINE void hknpCharacterProxy::setMaxSlope(hkReal maxSlope)
{
	m_maxSlopeCosine = hkMath::cos(maxSlope);
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
