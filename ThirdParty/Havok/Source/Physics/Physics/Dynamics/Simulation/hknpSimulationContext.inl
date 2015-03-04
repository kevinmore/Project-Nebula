/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

HK_FORCE_INLINE int hknpSimulationContext::getNumCpuThreads() const
{
	return m_numCpuThreads;
}

HK_FORCE_INLINE int hknpSimulationContext::getNumSpuThreads() const
{
	return m_numSpuThreads;
}

HK_FORCE_INLINE int hknpSimulationContext::getNumThreads() const
{
	return m_numCpuThreads + m_numSpuThreads;
}

HK_FORCE_INLINE hknpSimulationThreadContext* hknpSimulationContext::getThreadContext( int threadIdx )
{
	HK_ON_CPU( HK_ASSERT( 0xf0fdad10, threadIdx >=0 && threadIdx <= getNumThreads() ) );
	return &m_threadContexts[threadIdx];
}

#if defined(HK_PLATFORM_HAS_SPU)

HK_FORCE_INLINE void** hknpSimulationContext::getShapeVTablesPpu()
{
	return m_shapeVTablesPpu;
}

#endif	// !HK_PLATFORM_HAS_SPU

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
