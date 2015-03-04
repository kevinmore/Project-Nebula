/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE void hknpBreakableConstraintUtil::createConstraint
	(hknpConstraint& instance, hknpBodyId bodyIdA, hknpBodyId bodyIdB, hkpConstraintData* data, hkReal breakingThreshold)
{
	instance.initExportable( bodyIdA, bodyIdB, data, sizeof(hkReal) );
	instance.m_flags.orWith(hknpConstraint::RAISE_CONSTRAINT_FORCE_EXCEEDED_EVENTS);
	setBreakingThreshold( instance, breakingThreshold );
}

HK_FORCE_INLINE hkBool hknpBreakableConstraintUtil::isBreakable(hknpConstraint& instance)
{
#ifdef HK_DEBUG
	hkpConstraintData::RuntimeInfo info;
	instance.m_data->getRuntimeInfo(true, info);
	bool hasBreakableData = (instance.m_runtimeSize > info.m_sizeOfExternalRuntime);
	bool hasFlag = instance.m_flags.get(hknpConstraint::RAISE_CONSTRAINT_FORCE_EXCEEDED_EVENTS);
	HK_ASSERT(0x495610fd, !hasBreakableData == !hasFlag);
#endif
	return instance.m_flags.get(hknpConstraint::RAISE_CONSTRAINT_FORCE_EXCEEDED_EVENTS);
}

HK_FORCE_INLINE hkReal hknpBreakableConstraintUtil::getBreakingThreshold(hknpConstraint& instance)
{
	HK_ASSERT2(0x495610fd, hknpBreakableConstraintUtil::isBreakable(instance),
	"Breakable constraints must be created using hknpBreakingConstraintUtil::createConstraint()");
	hkReal* threshold = (hkReal*) hkAddByteOffset( instance.m_runtime, instance.m_runtimeSize - sizeof(hkReal) );
	return *threshold;
}

HK_FORCE_INLINE void hknpBreakableConstraintUtil::setBreakingThreshold(hknpConstraint& instance, hkReal value)
{
	HK_ASSERT2(0x7c16fca3, isBreakable(instance),
	"Breakable constraints must be created using hknpBreakingConstraintUtil::createConstraint()");
	hkReal* thresh = (hkReal*) hkAddByteOffset( instance.m_runtime, instance.m_runtimeSize - sizeof(hkReal) );
	*thresh = value;
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
