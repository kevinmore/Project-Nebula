/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_BOX_BOX_COLLISION_DETECTION_H
#define HK_COLLIDE2_BOX_BOX_COLLISION_DETECTION_H

#include <Physics2012/Collide/BoxBox/hkpBoxBoxManifold.h>

struct hkpProcessCollisionOutput;
struct hkpProcessCollisionInput;
class hkpCdBody;
class hkpBoxBoxManifold;
struct hkpProcessCdPoint;


// for a hkpFeatureContactPoint:
// case pointAfaceB or pointBfaceB
//              m_featureIdA hkpBoxBoxFeature index
//              m_featureIdB = 0x 0 [8 indicates -ve normalSign | 0 indicates +ve ] [axisMap] 0
//				where axisMap indicates how to traverse the box halfextent to get to the vertex = [0xyz] : x = 1 -> -halfextent.x 
// contents of m_featureIdA for an edge = 0x 0 0 [axisMap|Wbit] [edgeDirection]
// contents of m_featureIdB for an edge = 0x 0 0 [axisMap]      [edgeDirection]
// where edgeDirection = [0|1|2] to indicate if edge points in x,y,z direction. and axisMap gets start of edge
class hkpFeatureContactPoint;


class hkpBoxBoxCollisionDetection
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_CDINFO, hkpBoxBoxCollisionDetection);

		HK_FORCE_INLINE hkpBoxBoxCollisionDetection( const hkpCdBody& bodyA, const hkpCdBody& bodyB, const hkpProcessCollisionInput* env,
			hkpContactMgr* mgr,		hkpProcessCollisionOutput* result,
			const hkTransform &atb, 
			const hkTransform &wTa, hkVector4Parameter radiusA, 
			const hkTransform &wTb, hkVector4Parameter radiusB, hkSimdRealParameter tolerance );

		/// Find any collision points and add them to the manifold, remove any invalid ones
		void calcManifold( hkpBoxBoxManifold& manifold ) const;

		/// Return true if the boxes are overlapping, false if disjoint.
		/// sets the m_sepDist variables if returns true
		hkResult checkIntersection(const hkVector4& tolerance) const ;

		/// This public version also call checkIntersection internally !
		hkBool calculateClosestPoint( hkContactPoint& contact) const;


		inline hkBool getPenetrations();

	protected:
		/// Need to call before checkIntersection
		inline void initWorkVariables() const;

		// offer some control over performance vs quality.  When this is true additional edge-edge
		// collision points will be tested for when a new closest point is found.
		static hkBool m_attemptToFindAllEdges;

	protected:

		//!me move back to boxbox file....
		enum hkpBoxBoxFeature { 
			HK_BOXBOX_FACE_A_X = 0, HK_BOXBOX_FACE_A_Y = 1, HK_BOXBOX_FACE_A_Z = 2,
			HK_BOXBOX_FACE_B_X = 4, HK_BOXBOX_FACE_B_Y = 5, HK_BOXBOX_FACE_B_Z = 6,
			HK_BOXBOX_EDGE_0_0 = 8, HK_BOXBOX_EDGE_0_1 = 9, HK_BOXBOX_EDGE_0_2 =10,
			HK_BOXBOX_EDGE_1_0 =12, HK_BOXBOX_EDGE_1_1 =13, HK_BOXBOX_EDGE_1_2 =14,
			HK_BOXBOX_EDGE_2_0 =16, HK_BOXBOX_EDGE_2_1 =17, HK_BOXBOX_EDGE_2_2 =18,
			HK_BOXBOX_FEATURE_COUNT
		};

		static const hkReal m_manifoldConsistencyCheckAngularCosTolerance;
		static const hkReal m_coplanerAngularCosTolerance;
		static const hkReal m_contactNormalAngularCosTolerance;
		static const hkReal m_edgeEndpointTolerance;
		static const int m_maxFeaturesToReject;

		class hkpFeaturePointCache
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CDINFO, hkpBoxBoxCollisionDetection::hkpFeaturePointCache );

			// axis map vector in this space
			hkVector4 m_vA;
			hkVector4 m_vB;
			hkVector4 m_nA;

			// this is true if distance along absolute normal from this body to other is -ve
			hkVector4Comparison m_normalIsFlipped;

			hkSimdReal m_distance;
			int m_featureIndexA;
			int m_featureIndexB;
		};

		enum{ HK_FINDCLOSESTPOINT_NO_VALID_POINT, HK_FINDCLOSESTPOINT_POINT_IN_MANIFOLD, HK_FINDCLOSESTPOINT_VALID_POINT };


		typedef void (HK_CALL hkpBoxBoxCollisionDetection::*tFcnValidationDataFromFeatureId)( hkpFeaturePointCache&, const hkpFeatureContactPoint& ) const;
		typedef hkBool (HK_CALL hkpBoxBoxCollisionDetection::*tFcnIsValid)( hkpFeaturePointCache &) const;

	

		// only call after checkIntersecion has been called in a frame
		int findClosestPoint( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint& fcp, hkpFeaturePointCache& fpp ) const ;

		// update manifold points. remove some if necessary
		// only call after findClosestPoint has been called in a frame
		// returns true if the closest point must be rechecked before this method is called again
		HK_FORCE_INLINE void refreshManifold( hkpBoxBoxManifold& manifold, hkSimdReal& minContactPointDistance ) const;

		void tryToAddPointFaceA(	hkpBoxBoxManifold& manifold, const hkpFeatureContactPoint fcpTemplate, hkUint16 planeMask, hkSimdRealParameter closestPointDist ) const;
		void tryToAddPointFaceB(	hkpBoxBoxManifold& manifold, const hkpFeatureContactPoint fcpTemplate, hkUint16 planeMask, hkSimdRealParameter closestPointDist ) const;


		void tryToAddPointOnEdge( hkpBoxBoxManifold& manifold, int edgeA, int edgeB, int nextVertA, int nextVertB, const hkVector4& normalA, const hkVector4& normalB, hkSimdRealParameter closestPointDist ) const;

		void addAdditionalEdgeHelper( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint& fcp, hkSimdRealParameter closestPointDist ) const;

		HK_FORCE_INLINE void findAdditionalManifoldPoints( hkpBoxBoxManifold& manifold, hkpFeatureContactPoint fcp ) const;

		// bitSet Operations
	//	inline hkUint16 featureIndexFromFeatureId( hkUint16 fIA, hkUint16 fIB ) const;

		HK_FORCE_INLINE void faceAVertexBValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const; 
		HK_FORCE_INLINE void faceBVertexAValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const; 
		HK_FORCE_INLINE void     edgeEdgeValidationDataFromFeatureId( hkpFeaturePointCache& fpp, const hkpFeatureContactPoint &fcp ) const;


		HK_FORCE_INLINE void faceAVertexBContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const;
		HK_FORCE_INLINE void faceBVertexAContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const;
		HK_FORCE_INLINE void     edgeEdgeContactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const;
		
		HK_FORCE_INLINE void contactPointFromFeaturePointCache( hkpProcessCdPoint& ccpOut, const hkpFeatureContactPoint &fcp, const hkpFeaturePointCache& fpp ) const;

		HK_FORCE_INLINE void faceAVertexBValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp, int featureIndex ) const;
		HK_FORCE_INLINE void faceBVertexAValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp, int featureIndex ) const;
		HK_FORCE_INLINE void     edgeEdgeValidationDataFromFeatureIndex( hkpFeaturePointCache &fpp ) const;


		template <int edgeNext, int edgeNextNext>
		HK_FORCE_INLINE void setvdProj( const hkRotation& bRa, hkVector4& vdproj ) const;
		HK_FORCE_INLINE hkBool32 isValidFaceAVertexB( const hkpFeaturePointCache &fpp ) const;
		HK_FORCE_INLINE hkBool32 isValidFaceBVertexA( const hkpFeaturePointCache &fpp ) const;
		HK_FORCE_INLINE void calcDistanceFaceAVertexB( hkpFeaturePointCache& fpp ) const;
		HK_FORCE_INLINE void calcDistanceFaceBVertexA( hkpFeaturePointCache& fpp ) const;

		bool isValidEdgeEdge( hkpFeaturePointCache &fpp ) const;

		HK_FORCE_INLINE void calcManifoldNormal( hkVector4& manifoldN, const hkpFeatureContactPoint& fcp, hkpFeaturePointCache &fpp, bool isPointValidated ) const;
		
		HK_FORCE_INLINE hkBool32 queryManifoldNormalConsistency( hkpBoxBoxManifold& manifold ) const;
		HK_FORCE_INLINE void checkManifoldNormalConsistency( hkpBoxBoxManifold& manifold ) const;

		void checkCompleteness( hkpBoxBoxManifold& manifold, int planeMaskA, int planeMaskB ) const;

		// add a point to the manifold and result.
		HK_FORCE_INLINE hkResult addPoint( hkpBoxBoxManifold& manifold, const hkpFeaturePointCache& fpp, hkpFeatureContactPoint& fcp ) const;
		inline void removePoint( hkpBoxBoxManifold& manifold, int index ) const;

		// info for box we operate on
		const hkpCdBody& m_bodyA;
		const hkpCdBody& m_bodyB;
		const hkpProcessCollisionInput* m_env;

		// only needed for process collision
		hkpContactMgr*	m_contactMgr;
		hkpProcessCollisionOutput* m_result;

		const hkTransform& m_wTa;
		const hkTransform& m_wTb;
		hkTransform m_aTb;   // rotation ( basis change ) from b to a

		hkVector4 m_radiusA;		// the extents of the box
		hkVector4 m_radiusB;
		hkVector4 m_tolerance4;		// .w is forced to FLT_MAX
		hkVector4 m_keepRadiusA;	// the extents before contact points are getting deleted, this is m_radiusA + m_tolerance
		hkVector4 m_keepRadiusB;

		hkSimdReal m_boundaryTolerance;

		// these values are all calculated in checkIntersection

		mutable hkVector4 m_dinA;	// center from a to b in A's space
		mutable hkVector4 m_dinB;	// center from a to b in B's space

		// distance a's extent overlaps b's projected along feature axis
		// note the .w components are not used
		mutable hkVector4 m_sepDist[5]; 


	#ifdef HK_DEBUG
		void debugCheckManifold( hkpBoxBoxManifold& manifold, hkpProcessCollisionOutput* env ) const;
	#endif

};

#include <Physics2012/Collide/BoxBox/hkpBoxBoxCollisionDetection.inl>

#endif // HK_COLLIDE2_BOX_BOX_COLLISION_DETECTION_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
