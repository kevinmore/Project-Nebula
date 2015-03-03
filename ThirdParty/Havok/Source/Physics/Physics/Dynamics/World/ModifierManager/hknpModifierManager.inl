/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hkBool32 hknpModifierManager::isFunctionRegistered(
	hknpModifier::FunctionType functionType, hknpModifierFlags filter ) const
{
	const ModifierEntries& entries = m_modifiersPerFunction[ functionType ];
	return entries.m_allEnablingFlags & filter;
}

HK_FORCE_INLINE hknpBodyFlags hknpModifierManager::getCombinedBodyFlags(
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB ) const
{
	hknpBodyFlags bodyFlags     = cdBodyA.m_body->m_flags.get() | cdBodyB.m_body->m_flags.get();
	hknpBodyFlags materialFlags = cdBodyA.m_material->m_flags.get() | cdBodyB.m_material->m_flags.get();

	HK_ASSERT2( 0xf0dccfcc, 0 == (bodyFlags & ~hknpBody::FLAGS_MASK), "Illegal flag set on hknpBody" );
	HK_ASSERT2( 0xf0dccfcd, 0 == (materialFlags & ~hknpMaterial::FLAGS_MASK), "Illegal flag set on hknpMaterial" );

	hknpBodyFlags enabledModifiers = bodyFlags | materialFlags | m_globalBodyFlags;
	return enabledModifiers;
}

HK_FORCE_INLINE hknpBodyFlags hknpModifierManager::getCombinedBodyRootFlags(
	const hknpMaterial* materials, const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB )
{
	const hknpMaterial* matA = &materials[cdBodyA.m_body->m_materialId.value()];
	const hknpMaterial* matB = &materials[cdBodyB.m_body->m_materialId.value()];

	hknpBodyFlags bodyFlags = cdBodyA.m_body->m_flags.get() | cdBodyB.m_body->m_flags.get();
	hknpBodyFlags materialFlags = matA->m_flags.get() | matB->m_flags.get();

	HK_ASSERT2( 0xf0dccfcc, 0 == (bodyFlags & ~hknpBody::FLAGS_MASK), "Illegal flag set on hknpBody" );
	HK_ASSERT2( 0xf0dccfcd, 0 == (materialFlags & ~hknpMaterial::FLAGS_MASK), "Illegal flag set on hknpMaterial" );

	hknpBodyFlags enabledModifiers = bodyFlags | materialFlags | m_globalBodyFlags;
	return enabledModifiers;
}

HK_FORCE_INLINE hknpCollisionFilter* hknpModifierManager::getCollisionFilter() const
{
	return m_collisionFilter;
}

HK_FORCE_INLINE hknpCollisionFilter* hknpModifierManager::getCollisionQueryFilter() const
{
	return m_collisionQueryFilter;
}

HK_FORCE_INLINE hknpBodyFlags hknpModifierManager::getGlobalBodyFlags() const
{
	return m_globalBodyFlags;
}

HK_FORCE_INLINE void hknpModifierManager::fireWeldingModifier(
	const hknpSimulationThreadContext & tl, const hknpModifierSharedData& sharedData,
	hknpBodyQuality::Flags flags, const hknpCdBody& cdBodyA, hknpCdBody* cdBodyB,
	hknpManifold* manifolds, int numManifolds )
{
	hknpModifierManager* modifierMgr = this;
	if( flags.anyIsSet(hknpBodyQuality::ANY_WELDING) )
	{
		hknpWeldingModifier* weldingModifier = HK_NULL;
		while(1)
		{
			if( flags.anyIsSet(hknpBodyQuality::ENABLE_NEIGHBOR_WELDING) )
			{
				// neighbor welding can utilize motion welding on top of neighbor welding, so use it.
				weldingModifier = modifierMgr->m_neighborWeldingModifier;
			#if defined(HK_PLATFORM_SPU)
				HK_WARN_ONCE(0x3e7d7871, "Neighbor welding is not supported on SPU. Using motion welding instead");
				weldingModifier = modifierMgr->m_motionWeldingModifier;
			#endif
				break;
			}
			if( flags.anyIsSet(hknpBodyQuality::ENABLE_MOTION_WELDING) )
			{
				weldingModifier = modifierMgr->m_motionWeldingModifier;
				break;
			}
			if( flags.anyIsSet(hknpBodyQuality::ENABLE_TRIANGLE_WELDING) )
			{
				weldingModifier = modifierMgr->m_triangleWeldingModifier;
			#if defined(HK_PLATFORM_SPU)
				HK_WARN_ONCE(0x3e7d7871, "Triangle welding is not supported on SPU. Using motion welding instead");
				weldingModifier = modifierMgr->m_motionWeldingModifier;
			#endif
				break;
			}
			break;
		}

		hknpWeldingModifier::WeldingInfo wInfo;
		wInfo.m_qualityFlags = flags;
		weldingModifier->postMeshCollideCallback( tl, sharedData, wInfo, cdBodyA, *cdBodyB, manifolds, numManifolds );
	}
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
