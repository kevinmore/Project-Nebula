/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxMeshSectionUtil.h>
#include <Common/SceneData/Mesh/hkxVertexAnimation.h>

hkxVertexAnimationStateCache::hkxVertexAnimationStateCache(hkxVertexBuffer* b, bool alterInPlace)
: m_alteredVerts(b->getNumVertices(), hkBitFieldValue::ZERO), m_ownState(!alterInPlace), m_curKeyTime(0)
{
  m_state = HK_NULL;
  reset(b,alterInPlace);
}

hkxVertexAnimationStateCache::hkxVertexAnimationStateCache(const hkxVertexBuffer* b)
: m_alteredVerts(b->getNumVertices(), hkBitFieldValue::ZERO), m_ownState(true), m_curKeyTime(0)
{
  m_state = HK_NULL;
  copyState(b);
}

void hkxVertexAnimationStateCache::copyState(const hkxVertexBuffer* vb)
{
  if (!m_state)
  {
    m_state = new hkxVertexBuffer();
    m_state->setNumVertices( vb->getNumVertices(), vb->getVertexDesc() ); // same fmt, same num verts
  }
  m_state->copy(*vb, false); // memcpy
}

void hkxVertexAnimationStateCache::reset( hkxVertexBuffer* vb, bool alterInPlace)
{
  if (alterInPlace)
  {
	if (m_state) m_state->removeReference();
    m_state = vb;
    m_state->addReference();
  }
  else
  {
    copyState(vb);
  }
}

hkxVertexAnimationStateCache::~hkxVertexAnimationStateCache()
{
  m_state->removeReference();
}

// will take all new vert data from anim and alter the cached vb with it
void hkxVertexAnimationStateCache::apply(const hkxVertexAnimation* anim, float dt)
{
  const hkxVertexBuffer& partialVb = anim->m_vertData;
  const hkxVertexDescription& partialDecl = partialVb.getVertexDesc(); 
  const hkxVertexDescription& fullDecl = m_state->getVertexDesc(); 

  hkArray< const hkxVertexDescription::ElementDecl* > partialElems;
  hkArray< const hkxVertexDescription::ElementDecl* > fullElems;
  hkArray< hkUint32> perElemSize;

  hkArray< const hkUint8 * > partialPtrs;
  hkArray< hkUint8* > fullPtrs;

  int numAnimatedComponents = anim->m_componentMap.getSize();
  for (int ci=0; ci < numAnimatedComponents; ++ci)
  {
    const hkxVertexAnimation::UsageMap& m = anim->m_componentMap[ci];
    fullElems.pushBack( fullDecl.getElementDecl( m.m_use, m.m_useIndexOrig ) );
    partialElems.pushBack( partialDecl.getElementDecl( m.m_use, m.m_useIndexLocal ) );

    // to limit amount of code, but if happens can handle these:
    HK_ASSERT(0x7b36ada4, fullElems.back()->m_usage ==  partialElems.back()->m_usage );
    HK_ASSERT(0x43335461, fullElems.back()->m_type ==  partialElems.back()->m_type );
    HK_ASSERT(0x1c3f2742, fullElems.back()->m_numElements ==  partialElems.back()->m_numElements );

    partialPtrs.pushBack( (hkUint8*)partialVb.getVertexDataPtr( *partialElems.back() ));
    fullPtrs.pushBack( (hkUint8*)m_state->getVertexDataPtr( *fullElems.back() ));
    hkUint32 byteSize = fullDecl.getByteSizeForType( fullElems.back()->m_type, fullElems.back()->m_numElements );
    HK_ASSERT(0xe720dd2, byteSize > 0); 
    perElemSize.pushBack( byteSize );

#ifdef HK_DEBUG
    // Only used interpolate for pos and normal data so far, but feel free to add scalar floats below too
    if (dt > 0.f)
    {
      HK_ASSERT(0x38d35e48, partialElems[ci]->m_type == hkxVertexDescription::HKX_DT_FLOAT);
      HK_ASSERT(0x7049d6c6, partialElems[ci]->m_numElements >= 3);
    }
#endif
  }

  hkSimdReal dtS; dtS.setFromFloat(dt);
  hkBool32 interpolate = dtS.isGreaterZero();
  for (int vi=0; vi < partialVb.getNumVertices(); ++vi)
  {
    int origVertIndex = anim->m_vertexIndexMap[vi];
    HK_ASSERT(0x16ad00ee, (origVertIndex>=0) && (origVertIndex < m_state->getNumVertices() ) );
    
    m_alteredVerts.set( origVertIndex );

    for (int ci=0; ci < numAnimatedComponents; ++ci)
    {
       // asserts above should mean only difference is stride 
       const void* src = partialPtrs[ci] + (vi*partialElems[ci]->m_byteStride);
       void* dest = fullPtrs[ci] + (origVertIndex*fullElems[ci]->m_byteStride);
       if (interpolate)
       {
         // interpolate (we assume 4*hkFloat32 here for ease)
         hkVector4 srcV; srcV.load<4,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)src);
         hkVector4 destV; destV.load<4,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)dest);
         destV.setInterpolate(destV, srcV, dtS); // dest is cur cache state (prev frame say). Src is next frame, of which we want to get dt towards
		 destV.store<4,HK_IO_NATIVE_ALIGNED>((hkFloat32*)dest);
       }
       else
       {
         hkString::memCpy(dest, src,  perElemSize[ci] );
       }
    }
  }
  m_curKeyTime = anim->m_time; 
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
