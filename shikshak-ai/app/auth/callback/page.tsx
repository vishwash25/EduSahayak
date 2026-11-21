"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";

export default function AuthCallbackPage() {
  const router = useRouter();
  const started = useRef(false);
  const [message, setMessage] = useState("Completing Google sign-in...");

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const completeSignIn = async () => {
      try {
        const params = new URLSearchParams(window.location.search);
        const code = params.get("code");
        const authError = params.get("error_description") || params.get("error");

        if (authError) {
          throw new Error(authError);
        }

        const { data: sessionData } = await supabase.auth.getSession();

        if (!sessionData.session && code) {
          const { error: exchangeError } =
            await supabase.auth.exchangeCodeForSession(code);

          if (exchangeError) throw exchangeError;
        }

        const { data: finalSession } = await supabase.auth.getSession();

        if (!finalSession.session) {
          throw new Error(
            "Google sign-in completed, but no active session was found. Please try again."
          );
        }

        const { data, error: userError } = await supabase.auth.getUser();

        if (userError) throw userError;
        if (!data.user) throw new Error("Could not retrieve your account.");

        const existingRole = data.user.user_metadata?.role;

        if (existingRole && existingRole !== "teacher") {
          await supabase.auth.signOut();
          throw new Error(
            "This account is not registered as a Teacher. The Student dashboard is not available yet."
          );
        }

        if (!existingRole) {
          const { error: updateError } = await supabase.auth.updateUser({
            data: { role: "teacher" },
          });

          if (updateError) throw updateError;
        }

        setMessage("Sign-in successful. Redirecting...");
        router.replace("/");
      } catch (error) {
        console.error("Google authentication error:", error);
        setMessage(
          error instanceof Error
            ? error.message
            : "Something went wrong during Google sign-in."
        );
      }
    };

    completeSignIn();
  }, [router]);

  return (
    <main
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "#090d18",
        color: "#ffffff",
        padding: "24px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div style={{ textAlign: "center", maxWidth: "440px" }}>
        <h2 style={{ marginBottom: "14px" }}>EduSahayak</h2>
        <p style={{ color: "#cbd5e1", lineHeight: 1.6 }}>{message}</p>
      </div>
    </main>
  );
}